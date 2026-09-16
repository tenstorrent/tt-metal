# SPDX-License-Identifier: Apache-2.0
"""Olmo 3 tool-call parser that tolerates a missing ``<function_calls>`` wrapper.

Loaded by vLLM through ``--tool-parser-plugin <this file>`` and selected with
``--tool-call-parser olmo3_tt`` (see tt-model.yaml ``serve.capabilities`` / ``serve.args``).

Why this exists: vLLM's stock ``olmo3`` streaming parser decides on the very first
delta that the turn is plain text unless the accumulated text starts with ``<``.
Under agentic system prompts that demand minimal output (OpenCode's, for one),
Olmo-3.1-32B-Instruct regularly emits the bare pythonic call

    read(filePath="./secret.txt", limit=1)

with no ``<function_calls>`` opening tag (sometimes with the closing tag only), so
the whole call streams to the client as ``content`` and the agent never sees a tool
call (campaign T7, 2026-09-11). The non-streaming path has the same blind spot when
only the closing tag is present.

This subclass keeps the stock behaviour for tagged output and adds one rule: text
that starts with a *declared* tool name immediately followed by ``(`` is a tool call.
While the text is still a strict prefix of ``<name>(`` it is withheld (it may still
turn into prose such as ``read the file``); the moment it can no longer be a call, the
withheld text is flushed as content. Everything after the decision is delegated to
the stock parser with the tags normalised, so argument streaming, ids and
``finish_reason=tool_calls`` behave exactly as upstream.
"""

from __future__ import annotations

import regex as re
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, ExtractedToolCallInformation
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.olmo3_tool_parser import Olmo3PythonicToolParser

logger = init_logger(__name__)

OPEN_TAG = "<function_calls>"
CLOSE_TAG = "</function_calls>"
_IDENT = re.compile(r"^[A-Za-z_]\w*$")


def _tool_names(request, fallback) -> list[str]:
    names: list[str] = []
    tools = getattr(request, "tools", None) or fallback or []
    for tool in tools:
        fn = getattr(tool, "function", None)
        name = getattr(fn, "name", None) if fn is not None else getattr(tool, "name", None)
        if isinstance(name, str) and _IDENT.match(name):
            names.append(name)
    return names


class Olmo3LenientToolParser(Olmo3PythonicToolParser):
    """``olmo3`` parser + untagged-call detection (see module docstring)."""

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        # None: undecided (withholding), "tool": streaming a tool call, "text": plain content
        self._mode: str | None = None
        self._eos_ids = set()
        for tok in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    self._eos_ids.add(tid)
            except Exception:  # noqa: BLE001 - tokenizer without that token
                pass
        eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            self._eos_ids.add(eos)

    # ----- non-streaming -------------------------------------------------------------
    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        result = super().extract_tool_calls(model_output, request)
        if result.tools_called:
            return result
        # Stock parser only strips a complete <function_calls>…</function_calls> pair;
        # strip stray/unbalanced tags and retry once.
        body = model_output.strip()
        changed = False
        if body.startswith(OPEN_TAG):
            body, changed = body[len(OPEN_TAG) :], True
        if body.endswith(CLOSE_TAG):
            body, changed = body[: -len(CLOSE_TAG)], True
        if not changed:
            return result
        retry = super().extract_tool_calls(body.strip(), request)
        if retry.tools_called:
            return retry
        return result

    # ----- streaming -----------------------------------------------------------------
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ) -> DeltaMessage | None:
        if self._mode == "text":
            return DeltaMessage(content=delta_text)

        stripped = current_text.lstrip()
        finished = any(t in self._eos_ids for t in (delta_token_ids or ()))

        if self._mode is None:
            if stripped.startswith("<"):
                self._mode = "tool"
            else:
                names = _tool_names(request, self.tools)
                if any(stripped.startswith(n + "(") for n in names):
                    self._mode = "tool"
                elif stripped and not finished and any((n + "(").startswith(stripped) for n in names):
                    return None  # still a strict prefix of "<name>(": withhold, decide later
                elif not stripped and not finished:
                    return None  # only whitespace so far
                else:
                    self._mode = "text"
                    return DeltaMessage(content=current_text)  # flush everything withheld

        # tool mode: normalise to the tagged form the stock parser expects
        body = stripped
        if body.startswith(OPEN_TAG):
            body = body[len(OPEN_TAG) :]
        elif body.startswith("<") and OPEN_TAG.startswith(body):
            return None  # partial opening tag
        if body.endswith(CLOSE_TAG):
            body = body[: -len(CLOSE_TAG)]
        return super().extract_tool_calls_streaming(
            previous_text,
            OPEN_TAG + body,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )


ToolParserManager.register_module("olmo3_tt", module=Olmo3LenientToolParser)
