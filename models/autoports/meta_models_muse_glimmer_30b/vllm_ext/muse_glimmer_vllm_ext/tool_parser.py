# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Muse-Glimmer-30B.

The checkpoint's chat template emits its own function-call grammar, and **no
stock vLLM tool parser reads it**.  Without this, `tool_choice: "auto"` returns
`finish_reason=stop` with the raw block sitting in `content`: the model emits a
perfectly good tool call and the server never extracts it, which breaks every
agentic scaffold silently rather than loudly.

The grammar, quoted from the template's own instructions to the model::

    <atem:function_calls>
    <atem:invoke name="$FUNCTION_NAME">
    <atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
    ...
    </atem:invoke>
    </atem:function_calls>

Three properties of that grammar drive this implementation, all stated by the
template rather than inferred:

* **"The output is not expected to be valid XML and is parsed with regular
  expressions."**  So this is regex-based on purpose; an XML parser would reject
  output the model is explicitly licensed to produce.
* **"String and scalar parameters should be specified as is, while lists and
  objects should use JSON format."**  Parameters arrive as name/value text, not
  as a JSON object, so the object handed to the client has to be reassembled and
  each value decoded individually.
* **"Note that spaces for string values are not stripped."**  Values are
  therefore preserved verbatim, including a trailing newline when the model puts
  the closing tag on its own line.  See :func:`_decode_value`.

Function names may be namespaced (``example_tool.example_function``); the
template says to invoke the bare name when no namespace applies.  Names are
passed through untouched either way - resolving namespaces is the scaffold's job.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

logger = init_logger(__name__)

BLOCK_OPEN = "<atem:function_calls>"
BLOCK_CLOSE = "</atem:function_calls>"

_BLOCK_RE = re.compile(re.escape(BLOCK_OPEN) + r"(?P<body>.*?)" + re.escape(BLOCK_CLOSE), re.DOTALL)
_INVOKE_RE = re.compile(r'<atem:invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</atem:invoke>', re.DOTALL)
_PARAM_RE = re.compile(
    r'<atem:parameter\s+name="(?P<name>[^"]+)"\s*>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)

#: A value is JSON-decoded only when it *looks* like JSON. Feeding every value to
#: ``json.loads`` would coerce bare words the model meant as strings - and worse,
#: ``json.loads`` accepts ``NaN`` and ``Infinity``, so a parameter whose intended
#: string value is "NaN" would silently become a float.
_JSON_SHAPED = re.compile(r'^\s*(?:[{\["]|-?\d|true$|false$|null$)')


def _decode_value(raw: str):
    """Decode one parameter value.

    Verbatim unless it looks like JSON. The template states that spaces are not
    stripped, so nothing is trimmed here - a multi-line value whose closing tag
    sits on its own line keeps its trailing newline, which is what the model
    wrote. If a future checkpoint revision documents stripping, this is the one
    place to change.
    """
    if not _JSON_SHAPED.match(raw):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def parse_function_calls(text: str) -> list[tuple[str, dict]]:
    """Every ``(name, arguments)`` pair in ``text``, in emission order.

    A block may hold more than one ``<atem:invoke>``, which is how the model
    requests parallel calls; each becomes its own tool call.
    """
    calls: list[tuple[str, dict]] = []
    for block in _BLOCK_RE.finditer(text):
        for invoke in _INVOKE_RE.finditer(block.group("body")):
            args = {
                param.group("name"): _decode_value(param.group("value"))
                for param in _PARAM_RE.finditer(invoke.group("body"))
            }
            calls.append((invoke.group("name"), args))
    return calls


def _held_back(text: str) -> int:
    """Length of a trailing partial ``BLOCK_OPEN`` that must not be streamed yet.

    The open tag can be split across deltas. Emitting its first characters as
    content and only later discovering they began a tool call would leak markup
    to the client, so any suffix of ``text`` that is a proper prefix of the open
    tag is withheld.
    """
    for size in range(min(len(BLOCK_OPEN) - 1, len(text)), 0, -1):
        if text.endswith(BLOCK_OPEN[:size]):
            return size
    return 0


@ToolParserManager.register_module("muse_glimmer")
class MuseGlimmerToolParser(ToolParser):
    """Extracts ``<atem:function_calls>`` blocks into OpenAI tool calls."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._streamed_tool_calls = False

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        if BLOCK_OPEN not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        calls = parse_function_calls(model_output)
        if not calls:
            # An open tag with nothing parseable after it: an unterminated or
            # malformed block. Hand the text back rather than claim a tool call,
            # so the client sees the model's actual output.
            logger.warning("muse_glimmer: found %s but no complete invoke", BLOCK_OPEN)
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        # Prose before the first block is the assistant's message; anything after
        # the last block is discarded, matching the template's turn structure
        # where the call ends the assistant turn.
        prefix = model_output[: model_output.index(BLOCK_OPEN)]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[ToolCall(function=FunctionCall(name=name, arguments=json.dumps(args))) for name, args in calls],
            content=prefix if prefix.strip() else None,
        )

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
        """Stream content until a block opens, then emit the calls once complete.

        Deliberately does not stream partial arguments. Argument values here are
        opaque text that is only decodable once its closing tag arrives, so
        emitting fragments would mean publishing values that may still change.
        Clients see normal content streaming, then one delta carrying the
        finished tool calls.
        """
        if BLOCK_OPEN not in current_text:
            hold = _held_back(current_text)
            emit = current_text[: len(current_text) - hold]
            already = previous_text[: len(previous_text) - _held_back(previous_text)]
            new = emit[len(already) :] if emit.startswith(already) else emit
            return DeltaMessage(content=new) if new else None

        if self._streamed_tool_calls or BLOCK_CLOSE not in current_text:
            # Inside an open block, or already delivered: stay silent.
            return None

        calls = parse_function_calls(current_text)
        if not calls:
            return None

        self._streamed_tool_calls = True
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=json.dumps(args)),
                )
                for index, (name, args) in enumerate(calls)
            ]
        )
