# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

# Gemma4 tool-call wire format (emitted by the model, see the chat template):
#
#   <|tool_call>call:NAME{key:<|"|>str<|"|>,flag:true,n:3,xs:[1,2]}<tool_call|>
#
# Strings are wrapped in the ``<|"|>`` quote token; booleans render as
# ``true``/``false``; objects/arrays use ``{k:v,...}`` / ``[v,...]`` with
# *unescaped* keys; numbers are bare. This differs from FunctionGemma's
# ``<start_function_call>`` / ``<escape>`` format, so Gemma4 gets its own parser.
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
QUOTE = '<|"|>'
CALL_PREFIX = "call:"


class Gemma4ToolParser(ToolParser):
    """Tool parser for Google Gemma4 unified models.

    Parses the ``<|tool_call>``/``<tool_call|>`` delimited function-call format
    and normalizes arguments into a JSON string for the OpenAI tool-call schema.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Streaming state.
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = TOOL_CALL_START
        self.tool_call_end_token: str = TOOL_CALL_END

        # Buffer holding a possibly-partial trailing special token across
        # streaming deltas (the tokens span multiple pieces).
        self.buffered_delta_text: str = ""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        # The format is built entirely out of special tokens, so they must not
        # be stripped from the decoded output.
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------
    # Gemma4 argument-value grammar (recursive descent)
    # ------------------------------------------------------------------
    def _parse_value(self, s: str, i: int) -> tuple[Any, int]:
        """Parse one value starting at ``s[i]``; return (value, next_index)."""
        n = len(s)
        if s.startswith(QUOTE, i):
            inner_start = i + len(QUOTE)
            end = s.find(QUOTE, inner_start)
            if end == -1:
                return s[inner_start:], n
            return s[inner_start:end], end + len(QUOTE)

        c = s[i]
        if c == "{":
            return self._parse_object(s, i)
        if c == "[":
            return self._parse_array(s, i)

        # Bare token: number / bool / null (read up to the next delimiter).
        j = i
        while j < n and s[j] not in ",}]" and not s.startswith(QUOTE, j):
            j += 1
        return self._coerce_scalar(s[i:j].strip()), j

    def _parse_object(self, s: str, i: int) -> tuple[dict, int]:
        """Parse ``{k:v,...}`` starting at the opening brace ``s[i] == '{'``."""
        obj: dict[str, Any] = {}
        n = len(s)
        i += 1  # consume '{'
        while i < n:
            while i < n and s[i] in ", ":
                i += 1
            if i >= n or s[i] == "}":
                break
            colon = s.find(":", i)
            if colon == -1:
                break
            key = s[i:colon].strip()
            if (
                key.startswith(QUOTE)
                and key.endswith(QUOTE)
                and len(key) >= 2 * len(QUOTE)
            ):
                key = key[len(QUOTE) : -len(QUOTE)]
            value, i = self._parse_value(s, colon + 1)
            obj[key] = value
        if i < n and s[i] == "}":
            i += 1
        return obj, i

    def _parse_array(self, s: str, i: int) -> tuple[list, int]:
        """Parse ``[v,...]`` starting at the opening bracket ``s[i] == '['``."""
        arr: list[Any] = []
        n = len(s)
        i += 1  # consume '['
        while i < n:
            while i < n and s[i] in ", ":
                i += 1
            if i >= n or s[i] == "]":
                break
            value, i = self._parse_value(s, i)
            arr.append(value)
        if i < n and s[i] == "]":
            i += 1
        return arr, i

    @staticmethod
    def _coerce_scalar(token: str) -> Any:
        if token == "true":
            return True
        if token == "false":
            return False
        if token in ("null", "None", ""):
            return None
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            pass
        return token

    def _parse_call(self, block_body: str) -> tuple[str | None, dict]:
        """Parse a ``call:NAME{...}`` block body into (name, arguments)."""
        if not block_body.startswith(CALL_PREFIX):
            return None, {}
        brace = block_body.find("{", len(CALL_PREFIX))
        if brace == -1:
            return block_body[len(CALL_PREFIX) :].strip() or None, {}
        name = block_body[len(CALL_PREFIX) : brace].strip()
        arguments, _ = self._parse_object(block_body, brace)
        return (name or None), arguments

    def _iter_blocks(self, text: str):
        """Yield (block_body, complete) for each ``<|tool_call>`` block."""
        idx = 0
        n = len(text)
        while True:
            start = text.find(TOOL_CALL_START, idx)
            if start == -1:
                return
            body_start = start + len(TOOL_CALL_START)
            end = text.find(TOOL_CALL_END, body_start)
            if end == -1:
                yield text[body_start:n], False
                return
            yield text[body_start:end], True
            idx = end + len(TOOL_CALL_END)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if TOOL_CALL_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls: list[ToolCall] = []
            for body, complete in self._iter_blocks(model_output):
                if not complete:
                    continue
                name, arguments = self._parse_call(body)
                if not name:
                    continue
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            content_end = model_output.find(TOOL_CALL_START)
            content = model_output[:content_end].strip() if content_end > 0 else ""
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content or None,
            )
        except Exception:
            logger.exception("Error extracting Gemma4 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def _buffer_delta_text(self, delta_text: str) -> str:
        """Hold back a partial trailing special token until it completes."""
        combined = self.buffered_delta_text + delta_text
        for tag in (TOOL_CALL_START, TOOL_CALL_END, QUOTE):
            if combined.endswith(tag):
                self.buffered_delta_text = ""
                return combined
        for tag in (TOOL_CALL_START, TOOL_CALL_END, QUOTE):
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]
        self.buffered_delta_text = ""
        return combined

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        delta_text = self._buffer_delta_text(delta_text)
        current_text = previous_text + delta_text

        # No tool call yet: stream as plain content.
        if TOOL_CALL_START not in current_text:
            return DeltaMessage(content=delta_text) if delta_text else None

        try:
            start_count = current_text.count(TOOL_CALL_START)
            end_count = current_text.count(TOOL_CALL_END)
            prev_start_count = previous_text.count(TOOL_CALL_START)
            prev_end_count = previous_text.count(TOOL_CALL_END)

            # A new tool call opened in this delta: set up per-call state.
            if start_count > prev_start_count:
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})

            # The current tool call just closed: emit the complete arguments in a
            # single delta (concatenated deltas must form valid JSON, so we never
            # stream a partial object that already carries a closing brace).
            if end_count > prev_end_count:
                return self._emit_completed_call(current_text)

            # Mid open call: emit the function name as soon as it is known.
            if start_count > end_count and not self.current_tool_name_sent:
                return self._maybe_emit_name(current_text)

            return None
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _maybe_emit_name(self, current_text: str) -> DeltaMessage | None:
        start = current_text.rfind(TOOL_CALL_START) + len(TOOL_CALL_START)
        body = current_text[start:]
        if not body.startswith(CALL_PREFIX):
            return None
        func_part = body[len(CALL_PREFIX) :]
        if "{" not in func_part:
            return None
        func_name = func_part[: func_part.index("{")].strip()
        if not func_name:
            return None
        self.current_tool_name_sent = True
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(name=func_name).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _emit_completed_call(self, current_text: str) -> DeltaMessage | None:
        blocks = [
            body for body, complete in self._iter_blocks(current_text) if complete
        ]
        if not (0 <= self.current_tool_id < len(blocks)):
            return None
        if self.streamed_args_for_tool[self.current_tool_id]:
            return None  # already emitted
        name, arguments = self._parse_call(blocks[self.current_tool_id])
        args_json = json.dumps(arguments, ensure_ascii=False)
        self.streamed_args_for_tool[self.current_tool_id] = args_json
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": name,
            "arguments": arguments,
        }

        # If the whole call arrived in one delta the name was never sent; include
        # it (plus id/type) here. Otherwise emit only the arguments for this index.
        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            function = DeltaFunctionCall(name=name, arguments=args_json)
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=function.model_dump(exclude_none=True),
                    )
                ]
            )
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=args_json).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )
