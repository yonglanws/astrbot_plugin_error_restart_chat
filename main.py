"""astrbot_plugin_error_restart_chat

限制 LLM 单次回复的字数（字符数）：
- 回复超过配置的最大字数时，自动使用与原请求完全相同的配置
  （系统提示词、上下文、提示词、图片、模型等）重新生成；
- 超限的坏回复不会写入会话历史，也不会发送到聊天平台；
- 重试全部失败时可选择丢弃回复（发送提示）或照常发送原回复。

实现原理（基于 AstrBot v4 内置 Agent 流程）：
1. on_llm_request 钩子：在调用模型前对 ProviderRequest 做一份快照；
2. on_llm_response 钩子：检查回复纯文本长度，超限则用快照重放请求，
   成功后直接替换 LLMResponse 的 result_chain（此钩子触发于结果
   用于展示之前，因此用户看到的是替换后的回复）；
3. on_agent_done 钩子：AstrBot 落库时读取的是 run_context.messages，
   其中那条 assistant 消息在钩子触发前就已追加，因此这里需要同步
   修补其文本（或标记 _no_save 使其被跳过），坏回复才不会进入历史。
"""

import copy

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import (
    AstrMessageEvent,
    MessageChain,
    ResultContentType,
    filter,
)
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

# run_context.messages 中的消息为内部 Message/TextPart 类型，
# 做防御式导入：导入失败时仅影响历史修补，不影响检测与重试。
try:
    from astrbot.core.agent.message import TextPart
except ImportError:  # pragma: no cover
    TextPart = None

STATE_EXTRA_KEY = "_plugin_error_restart_chat_state"
"""挂在 event extra 上的状态键，用于在 on_llm_response / on_agent_done 间传递。"""

# 快照表上限：正常情况下同一会话的快照会被下一次请求覆盖，
# 此上限仅用于异常情况（如第三方 Agent 不触发响应钩子）时的防御性清理。
SNAPSHOT_MAX = 512

DEFAULT_FAIL_NOTICE = "⚠️ 本次回复超出字数限制，自动重试后仍失败，本次回复已被丢弃。"


@register(
    "astrbot_plugin_error_restart_chat",
    "mzk",
    "限制 LLM 单次回复字数，超限自动用相同配置重新生成，坏回复不写入会话历史",
    "v1.0.0",
    "https://github.com/yonglanws/astrbot_plugin_error_restart_chat",
)
class ErrorRestartChatPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # unified_msg_origin -> 请求快照（供超限重试时复刻原请求）
        self._req_snapshots: dict[str, dict] = {}

    # ---------------- 配置读取 ----------------

    def _int_cfg(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    @property
    def _max_output_chars(self) -> int:
        return self._int_cfg("max_output_chars", 500)

    @property
    def _max_retries(self) -> int:
        return max(0, self._int_cfg("max_retries", 2))

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _response_text(resp: LLMResponse) -> str:
        """取回复的纯文本（兼容 result_chain / completion_text 两种承载方式）。"""
        try:
            return resp.completion_text or ""
        except Exception:
            return ""

    @staticmethod
    def _is_streaming(event: AstrMessageEvent) -> bool:
        """通过事件结果上的流式标记判断当前是否处于流式输出模式。"""
        try:
            result = event.get_result()
            return bool(
                result
                and result.result_content_type == ResultContentType.STREAMING_RESULT
            )
        except Exception:
            return False

    def _snapshot_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """保存请求快照。contexts 做拷贝，防止 Agent 运行期间向原列表追加消息。"""
        try:
            try:
                contexts = copy.deepcopy(req.contexts)
            except Exception:
                contexts = list(req.contexts or [])
            self._req_snapshots[event.unified_msg_origin] = {
                "prompt": req.prompt,
                "contexts": contexts or [],
                "system_prompt": req.system_prompt,
                "image_urls": list(req.image_urls or []),
                "audio_urls": list(req.audio_urls or []),
                "extra_user_content_parts": list(req.extra_user_content_parts or []),
                "func_tool": req.func_tool,
                "tool_calls_result": req.tool_calls_result,
                "model": req.model,
                "session_id": req.session_id,
            }
            if len(self._req_snapshots) > SNAPSHOT_MAX:
                self._req_snapshots.clear()
        except Exception as e:
            logger.error(f"[error_restart_chat] 保存请求快照失败: {e}")

    async def _regenerate(self, event: AstrMessageEvent, snapshot: dict):
        """用与原请求相同的配置重新请求 LLM。"""
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            logger.warning("[error_restart_chat] 未找到当前使用的 LLM 提供商，无法重试")
            return None
        kwargs = dict(snapshot)
        hint = str(self.config.get("retry_hint") or "")
        if hint:
            base = snapshot.get("prompt") or ""
            kwargs["prompt"] = f"{base}{hint}" if base else hint
        return await provider.text_chat(**kwargs)

    @staticmethod
    def _patched_assistant_content(msg, new_text: str):
        """重建 assistant 消息内容：保留思考部分，正文替换为重试后的文本。"""
        content = msg.content
        if content is None or isinstance(content, str):
            return new_text
        parts = [p for p in content if getattr(p, "type", None) == "think"]
        parts.append(TextPart(text=new_text))
        return parts

    # ---------------- 钩子 ----------------

    @filter.on_llm_request(priority=-100)
    async def capture_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前：保存请求快照。

        priority=-100 使本钩子在（同类型钩子中）最后执行，
        尽量拿到其他插件修改完成后的最终请求配置。
        """
        self._snapshot_request(event, req)

    @filter.on_llm_response(priority=100)
    async def check_and_regenerate(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应后：检查字数，超限则用原配置重新生成并替换本次响应。

        priority=100 使本钩子先于其他响应处理类插件（TTS 转换等）执行，
        保证检测的是模型的原始文本，且替换后的内容继续走后续装饰流程。
        """
        state = {"action": "none", "text": ""}
        event.set_extra(STATE_EXTRA_KEY, state)

        try:
            limit = self._max_output_chars
            if limit <= 0:
                return
            # 仅处理纯文本的最终回复：工具调用轮次、流式分片、错误响应不处理
            if resp.role != "assistant" or resp.is_chunk:
                return
            if getattr(resp, "tools_call_name", None):
                return

            text = self._response_text(resp)
            if not text or len(text) <= limit:
                return

            snapshot = self._req_snapshots.pop(event.unified_msg_origin, None)
            if not snapshot:
                logger.warning(
                    "[error_restart_chat] 回复超限但未找到请求快照，跳过重试"
                )
                return

            logger.info(
                f"[error_restart_chat] 回复 {len(text)} 字，超过上限 {limit} 字，"
                f"开始重新生成（最多 {self._max_retries} 次）"
            )

            for attempt in range(1, self._max_retries + 1):
                try:
                    new_resp = await self._regenerate(event, snapshot)
                except Exception as e:
                    logger.error(
                        f"[error_restart_chat] 第 {attempt} 次重新生成请求失败: {e}"
                    )
                    new_resp = None
                if new_resp is None:
                    continue

                new_text = self._response_text(new_resp).strip()
                if (
                    new_resp.role == "assistant"
                    and not getattr(new_resp, "tools_call_name", None)
                    and new_text
                    and len(new_text) <= limit
                ):
                    # 直接替换响应内容：此钩子触发于结果用于展示之前，
                    # 后续发送与装饰均使用新内容。
                    resp.result_chain = MessageChain().message(new_text)
                    if getattr(new_resp, "raw_completion", None) is not None:
                        resp.raw_completion = new_resp.raw_completion
                    if getattr(new_resp, "usage", None) is not None:
                        try:
                            resp.usage = new_resp.usage
                        except Exception:
                            pass
                    state["action"] = "replaced"
                    state["text"] = new_text
                    logger.info(
                        f"[error_restart_chat] 第 {attempt} 次重试成功，"
                        f"新回复 {len(new_text)} 字"
                    )
                    # 流式模式下原回复已实时发出无法撤回，这里补发修正后的回复
                    if self._is_streaming(event):
                        state["action"] = "replaced_stream"
                        await event.send(MessageChain().message(new_text))
                    return
                logger.info(
                    "[error_restart_chat] "
                    f"第 {attempt} 次重试仍不合规（长度 {len(new_text) if new_text else 0}）"
                )

            # 重试全部失败
            if str(self.config.get("on_fail") or "discard") == "keep":
                state["action"] = "keep"
                logger.info("[error_restart_chat] 重试失败，按配置照常发送原回复")
                return

            state["action"] = "discard"
            # 先清空响应内容：即使个别版本仍尝试落库/发送，空回复也会被核心跳过
            resp.result_chain = MessageChain()
            notice = str(self.config.get("fail_notice") or DEFAULT_FAIL_NOTICE)
            try:
                await event.send(MessageChain().message(notice))
            finally:
                # 终止事件传播：坏回复不再进入发送与落库流程
                event.stop_event()
            logger.info(
                f"[error_restart_chat] 重试 {self._max_retries} 次均失败，"
                "已丢弃本次回复"
            )
        except Exception as e:
            logger.error(f"[error_restart_chat] 处理 LLM 响应时出错: {e}", exc_info=True)

    @filter.on_agent_done()
    async def patch_history(self, event: AstrMessageEvent, run_context, resp):
        """Agent 结束时：同步修正即将写入会话历史的 assistant 消息。

        AstrBot 落库读取 run_context.messages，其中超限的回复在本钩子
        触发前就已追加，因此必须在落库前修补或移除。
        """
        state = event.get_extra(STATE_EXTRA_KEY)
        self._req_snapshots.pop(event.unified_msg_origin, None)
        if not state or state.get("action") not in (
            "replaced",
            "replaced_stream",
            "discard",
        ):
            return
        if TextPart is None:
            logger.warning(
                "[error_restart_chat] 当前版本无法修补会话历史（缺少内部类型），"
                "本次回复可能仍会写入历史"
            )
            return
        try:
            for msg in reversed(run_context.messages):
                if getattr(msg, "role", None) == "assistant":
                    if state["action"] == "discard":
                        # 标记为不保存：核心落库时会跳过该条消息
                        msg._no_save = True
                    else:
                        msg.content = self._patched_assistant_content(
                            msg, state["text"]
                        )
                    break
        except Exception as e:
            logger.error(f"[error_restart_chat] 修正会话历史失败: {e}", exc_info=True)

    async def terminate(self):
        """插件卸载/停用时清理快照。"""
        self._req_snapshots.clear()
