# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OpenAI chat protocol for the Qwen3.8 server: request normalisation, tool-aware rendering, reply assembly.

Request side: the client's messages and tools are changed only where the
reference (``tokenizer.apply_chat_template`` on the raw request) needs help,
``arguments`` sent as a JSON string and ``content: null``; the tools render in
the client's key order, which ``tojson`` follows.  The device prompt of every
request is that reference render: a follow-up turn holds the served reply as
the template re-renders it (its content and tool calls, an empty think block),
never the recorded reasoning ids.  Reply side: generated token ids become
``reasoning_content``, ``content`` and ``tool_calls`` pieces in the order the
model emitted them, with the whitespace the template trims held back, so the
streamed and the non-streamed message agree.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import jinja2

from models.demos.blackhole.qwen38_flash_next.chat import (
    EOS_TOKEN_IDS,
    VOCAB_SIZE,
    Qwen38ChatFormatError,
    parse_tool_calls,
)

# tokenizer_config.json added_tokens_decoder; verified against the tokenizer by the checkpoint tests.
THINK_START_ID = 248_068
THINK_END_ID = 248_069
TOOL_CALL_START_ID = 248_058
TOOL_CALL_END_ID = 248_059
ROLES = ("system", "user", "assistant", "tool")
TOOL_CHOICES = ("auto", "none")
MAX_STOP_STRINGS = 4
FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")

# The client's reasoning_effort (OpenAI values plus the template's own) -> the template's level.
EFFORT_LEVELS = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}
# Reasoning tokens before </think> is forced, per level; None = only the answer reserve bounds it (xhigh is unbounded
# within the request's max_tokens, which defaults to the remaining context; the model card wants 32k of output room).
THINKING_TOKEN_CAPS = {"low": 4096, "medium": 16_384, "xhigh": None}
ANSWER_RESERVE_TOKENS = 256
# Image parts (OpenAI content-part shape): ``data:`` URLs only in this release (the server fetches nothing); the
# ``detail`` field is honoured (``low`` caps the pixel count, ``auto`` / ``high`` run the stock processor).  Video
# parts are refused in every spelling; so are images outside user messages and unknown part types.
IMAGE_DETAILS = ("auto", "low", "high")
IMAGE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp")
MAX_IMAGE_BYTES = 16 << 20  # decoded bytes of one image (the request body cap bounds the base64 form)
VIDEO_PART_SPELLINGS = ("video", "video_url", "input_video")
_DATA_URL = re.compile(
    r"^data:(?P<media>[a-z]+/[a-z0-9.+-]+)(?P<params>(?:;[a-z0-9-]+=[^;,]*)*);base64,(?P<payload>.*)$", re.S | re.I
)


@dataclass(frozen=True)
class Qwen38ImagePart:
    """One image of a request: the decoded bytes, their media type, the ``detail`` level and where the part sat."""

    data: bytes
    media_type: str
    detail: str
    message_index: int
    part_index: int


class Qwen38ChatRequestRejected(ValueError):
    """HTTP 400 with the OpenAI error fields; the message states actual vs expected."""

    def __init__(self, message: str, *, param: str | None = None, code: str = "bad_request") -> None:
        super().__init__(message)
        self.param = param
        self.code = code


# -- request side --------------------------------------------------------------------------------


def _image_part(item: Mapping[str, Any], where: str, *, message_index: int, part_index: int) -> Qwen38ImagePart:
    """An ``image_url`` part: ``{"type": "image_url", "image_url": {"url": "data:...;base64,...", "detail": ...}}``
    (or the string shorthand for ``image_url``); only ``data:`` URLs, an image media type, a decodable payload."""

    source = item.get("image_url")
    if isinstance(source, str):
        source = {"url": source}
    if not isinstance(source, Mapping) or not isinstance(source.get("url"), str):
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url must be {{'url': str, 'detail'?: str}} or the url string, got {source!r}",
            param=f"{where}.image_url",
        )
    detail = source.get("detail", "auto")
    if detail not in IMAGE_DETAILS:
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url.detail must be one of {IMAGE_DETAILS}, got {detail!r}",
            param=f"{where}.image_url.detail",
        )
    url = source["url"]
    if url[:5].lower() != "data:":
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url.url must be a data: URL (this server fetches no remote images), got {url[:24]!r}...",
            param=f"{where}.image_url.url",
        )
    match = _DATA_URL.match(url)
    if match is None or match.group("media").lower() not in IMAGE_MEDIA_TYPES:
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url.url must be data:<{'|'.join(IMAGE_MEDIA_TYPES)}>;base64,<payload>",
            param=f"{where}.image_url.url",
        )
    payload = match.group("payload")
    if len(payload) > (MAX_IMAGE_BYTES * 4) // 3 + 4:
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url.url decodes past the {MAX_IMAGE_BYTES} byte image limit", param=f"{where}.image_url.url"
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url.url is not valid base64: {error}", param=f"{where}.image_url.url"
        ) from error
    if not data:
        raise Qwen38ChatRequestRejected(f"{where}.image_url.url holds an empty image", param=f"{where}.image_url.url")
    if len(data) > MAX_IMAGE_BYTES:
        raise Qwen38ChatRequestRejected(
            f"{where}.image_url.url decodes to {len(data)} bytes, over the {MAX_IMAGE_BYTES} byte image limit",
            param=f"{where}.image_url.url",
        )
    return Qwen38ImagePart(data, match.group("media").lower(), detail, message_index, part_index)


def _content(
    value: Any, where: str, *, role: str, message_index: int, images: list[Qwen38ImagePart] | None
) -> str | list[dict[str, str]]:
    """A message's content for the template: a string, or a list of ``text`` items and (user messages, when the
    caller collects ``images``) ``image`` items, one per accepted ``image_url`` part, in order.  Video parts and
    unknown part types are refused, never dropped."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise Qwen38ChatRequestRejected(
            f"{where}.content must be a string, null or a list of content parts, got {type(value).__name__}",
            param=f"{where}.content",
        )
    items = []
    for index, item in enumerate(value):
        part = f"{where}.content[{index}]"
        if not isinstance(item, Mapping):
            raise Qwen38ChatRequestRejected(f"{part} must be a content part object, got {item!r}", param=part)
        kind = item.get("type")
        if kind in VIDEO_PART_SPELLINGS or any(key in item for key in VIDEO_PART_SPELLINGS):
            raise Qwen38ChatRequestRejected(f"{part}: video parts are not supported by this server", param=part)
        if kind == "text":
            if not isinstance(item.get("text"), str):
                raise Qwen38ChatRequestRejected(
                    f"{part} must be {{'type': 'text', 'text': str}}, got {item!r}", param=part
                )
            items.append({"type": "text", "text": item["text"]})
            continue
        if kind == "image_url":
            if images is None:
                raise Qwen38ChatRequestRejected(f"{part}: image parts are not accepted here (text-only)", param=part)
            if role != "user":
                raise Qwen38ChatRequestRejected(f"{part}: image parts belong to user messages, not {role}", param=part)
            images.append(_image_part(item, part, message_index=message_index, part_index=index))
            items.append({"type": "image"})
            continue
        raise Qwen38ChatRequestRejected(
            f"{part}.type must be 'text' or 'image_url', got {kind!r} (video and other parts are not supported)",
            param=part,
        )
    return items


def _text_content(value: Any, where: str) -> str | list[dict[str, str]]:
    """Text parts only (the callers that take no images)."""

    return _content(value, where, role="user", message_index=0, images=None)


def _tool_call(call: Any, where: str) -> dict[str, Any]:
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        raise Qwen38ChatRequestRejected(f"{where} must be {{'type': 'function', 'function': {{...}}}}", param=where)
    name = function.get("name")
    if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name):
        raise Qwen38ChatRequestRejected(
            f"{where}.function.name must match {FUNCTION_NAME.pattern}, got {name!r}", param=f"{where}.function.name"
        )
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as error:
            raise Qwen38ChatRequestRejected(
                f"{where}.function.arguments is not valid JSON: {error}", param=f"{where}.function.arguments"
            ) from error
    if not isinstance(arguments, Mapping):
        raise Qwen38ChatRequestRejected(
            f"{where}.function.arguments must be a JSON object (or its string), got {type(arguments).__name__}",
            param=f"{where}.function.arguments",
        )
    return {"type": "function", "function": {"name": name, "arguments": dict(arguments)}}


def normalize_messages(messages: Any, *, images: list[Qwen38ImagePart] | None = None) -> list[dict[str, Any]]:
    """The template's message schema from the client's: roles and order checked, ``content: null`` -> "",
    string ``arguments`` -> dict; ``id``, ``name`` and ``tool_call_id`` dropped (the template ignores them).
    ``images`` (a list the caller owns) collects the user messages' ``image_url`` parts as :class:`Qwen38ImagePart`
    in prompt order, each rendered as one ``image`` item (the template's one ``<|image_pad|>``); without it an image
    part is refused."""

    if not isinstance(messages, list) or not messages:
        raise Qwen38ChatRequestRejected(
            f"messages must be a nonempty list, got {type(messages).__name__}", param="messages"
        )
    result = []
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise Qwen38ChatRequestRejected(f"{where} must be an object, got {type(message).__name__}", param=where)
        role = message.get("role")
        if role not in ROLES:
            raise Qwen38ChatRequestRejected(f"{where}.role must be one of {ROLES}, got {role!r}", param=f"{where}.role")
        if role == "system" and index != 0:
            raise Qwen38ChatRequestRejected(
                f"{where}.role is system; the system message must be first", param=f"{where}.role"
            )
        copied: dict[str, Any] = {
            "role": role,
            "content": _content(message.get("content"), where, role=role, message_index=index, images=images),
        }
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise Qwen38ChatRequestRejected(
                        f"{where}.reasoning_content must be a string, got {type(reasoning).__name__}",
                        param=f"{where}.reasoning_content",
                    )
                copied["reasoning_content"] = reasoning
            calls = message.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    raise Qwen38ChatRequestRejected(
                        f"{where}.tool_calls must be a list, got {type(calls).__name__}", param=f"{where}.tool_calls"
                    )
                if calls:
                    copied["tool_calls"] = [
                        _tool_call(call, f"{where}.tool_calls[{position}]") for position, call in enumerate(calls)
                    ]
        result.append(copied)
    return result


def validate_tools(tools: Any) -> list[dict[str, Any]]:
    """The client's tool list checked by inspection and returned in its own key order (never re-sorted)."""

    if tools is None:
        return []
    if not isinstance(tools, list):
        raise Qwen38ChatRequestRejected(f"tools must be a list, got {type(tools).__name__}", param="tools")
    result = []
    for index, tool in enumerate(tools):
        where = f"tools[{index}]"
        function = tool.get("function") if isinstance(tool, Mapping) and tool.get("type") == "function" else None
        if not isinstance(function, Mapping):
            raise Qwen38ChatRequestRejected(f"{where} must be {{'type': 'function', 'function': {{...}}}}", param=where)
        name = function.get("name")
        if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name):
            raise Qwen38ChatRequestRejected(
                f"{where}.function.name must match {FUNCTION_NAME.pattern}, got {name!r}",
                param=f"{where}.function.name",
            )
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise Qwen38ChatRequestRejected(
                f"{where}.function.description must be a string, got {type(description).__name__}",
                param=f"{where}.function.description",
            )
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise Qwen38ChatRequestRejected(
                f"{where}.function.parameters must be a JSON-schema object, got {type(parameters).__name__}",
                param=f"{where}.function.parameters",
            )
        try:
            result.append(json.loads(json.dumps(tool, allow_nan=False)))
        except (TypeError, ValueError) as error:
            raise Qwen38ChatRequestRejected(f"{where} is not finite JSON: {error}", param=where) from error
    return result


def validate_stop(stop: Any) -> tuple[str, ...]:
    if stop is None:
        return ()
    strings = [stop] if isinstance(stop, str) else stop
    if (
        not isinstance(strings, list)
        or len(strings) > MAX_STOP_STRINGS
        or not all(isinstance(value, str) and value for value in strings)
    ):
        raise Qwen38ChatRequestRejected(
            f"stop must be a nonempty string or up to {MAX_STOP_STRINGS} of them, got {stop!r}", param="stop"
        )
    return tuple(strings)


def effort_level(value: Any) -> str:
    if not isinstance(value, str) or value not in EFFORT_LEVELS:
        raise Qwen38ChatRequestRejected(
            f"reasoning_effort must be one of {tuple(EFFORT_LEVELS)}, got {value!r}", param="reasoning_effort"
        )
    return EFFORT_LEVELS[value]


def thinking_budget(level: str, max_tokens: int, requested: int | None = None) -> int:
    """Reasoning tokens before </think> is forced: the client's ``thinking_budget``, else the level's cap; both
    leave ``ANSWER_RESERVE_TOKENS`` of ``max_tokens`` (at least half of it) for the answer after the forced
    ``</think>``, which counts against ``max_tokens`` too."""

    cap = THINKING_TOKEN_CAPS[level] if requested is None else requested
    room = max(max_tokens - ANSWER_RESERVE_TOKENS - 1, max_tokens // 2)
    return room if cap is None else min(cap, room)


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    enable_thinking: bool,
    reasoning_effort: str,
    add_generation_prompt: bool = True,
) -> str:
    """The checkpoint's template on normalised messages and validated tools (``preserve_thinking`` always true)."""

    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            preserve_thinking=True,
            reasoning_effort=reasoning_effort,
        )
    except jinja2.exceptions.TemplateError as error:
        raise Qwen38ChatRequestRejected(
            f"the chat template rejected the messages: {error}", param="messages"
        ) from error


def encode_chat(tokenizer: Any, text: str) -> list[int]:
    ids = [int(value) for value in tokenizer(text, add_special_tokens=False).input_ids]
    if not ids or min(ids) < 0 or max(ids) >= VOCAB_SIZE:
        raise Qwen38ChatFormatError(
            f"rendered chat has {len(ids)} ids, range {min(ids, default=None)}..{max(ids, default=None)}"
        )
    return ids


def render_prompt(
    tokenizer: Any,
    messages: Any,
    tools: Any,
    *,
    enable_thinking: bool,
    reasoning_effort: str,
    images: list[Qwen38ImagePart] | None = None,
    image_grids: Sequence[Any] = (),
) -> list[int]:
    """Prompt ids of a client request: normalise, validate, render with the generation prompt, encode; with images,
    the template's one ``<|image_pad|>`` per image is expanded to the grid's merged-token count
    (``mrope.expand_image_pads``; ``image_grids`` are the images' grids in prompt order, one per collected part)."""

    text = render_chat(
        tokenizer,
        normalize_messages(messages, images=images),
        validate_tools(tools),
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    if not text.endswith("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"):
        raise Qwen38ChatFormatError(f"official template generation suffix drifted: {text[-40:]!r}")
    ids = encode_chat(tokenizer, text)
    grids = list(image_grids)
    if grids or images:
        from models.demos.blackhole.qwen38_flash_next import mrope

        if images is not None and len(grids) != len(images):
            raise Qwen38ChatFormatError(f"{len(images)} image parts but {len(grids)} image grids")
        try:
            ids = mrope.expand_image_pads(ids, grids)
        except ValueError as error:
            raise Qwen38ChatFormatError(f"image pad expansion: {error}") from error
    return ids


# -- reply side ----------------------------------------------------------------------------------


class Qwen38ReplyAssembler:
    """Generated ids -> OpenAI message pieces, one ``push`` per token, ``finish`` at the end.

    Phases: reasoning (until 248069), content, tool (between 248058 and 248059,
    parsed as one block at its end, its arguments typed by ``tools``).  A tag id
    acts only in the phase that expects it and is text anywhere else: a tool
    block drafted inside the reasoning is reasoning, think tags quoted in the
    answer or inside a tool block are their text.  Held back: a split UTF-8
    sequence, the whitespace at each phase's edges (the template trims it), the
    tail that could start a stop string.  A block the parser rejects, or one cut
    by the token budget, is emitted as raw content; the caller logs
    ``parse_errors``.
    """

    def __init__(
        self,
        decode: Callable[[list[int]], str],
        *,
        thinking_open: bool,
        stop_strings: Sequence[str] = (),
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.decode = decode
        self.phase = "reasoning" if thinking_open else "content"
        self.stop_strings = tuple(stop_strings)
        self.tools = list(tools)
        self.stop_hold = max((len(value) for value in self.stop_strings), default=1) - 1
        self.pending_ids: list[int] = []
        self.held_text = ""
        self.phase_text_started = False
        self.tool_ids: list[int] = []
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.parse_errors: list[str] = []
        self.tokens = 0
        self.reasoning_tokens = 0
        self.stop_hit = False
        self.truncated_tool_call = False

    def push(self, token_id: int) -> list[dict[str, Any]]:
        if self.stop_hit or token_id in EOS_TOKEN_IDS:  # EOS reaches the assembler only with ignore_eos
            return []
        self.tokens += 1
        if self.phase == "tool":
            if token_id != TOOL_CALL_END_ID:
                self.tool_ids.append(token_id)
                return []
            block = "<tool_call>" + self.decode(self.tool_ids) + "</tool_call>"
            self.phase = "content"
            try:
                _visible, parsed = parse_tool_calls(block, self.tools)
            except Qwen38ChatFormatError as error:
                self.parse_errors.append(str(error))
                return self._text(block)
            calls = [
                {
                    "index": len(self.calls) + position,
                    "id": f"call_{secrets.token_hex(6)}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for position, call in enumerate(parsed)
            ]
            self.calls.extend(calls)
            return [{"tool_calls": calls}]
        if self.phase == "reasoning":
            if token_id == THINK_END_ID:
                return self._enter("content")
            self.reasoning_tokens += 1
        elif token_id == TOOL_CALL_START_ID:
            deltas = self._enter("tool")
            self.tool_ids = []
            return deltas
        self.pending_ids.append(token_id)
        text = self.decode(self.pending_ids)
        if text.endswith("�"):
            return []
        self.pending_ids = []
        return self._text(text)

    def finish(self) -> list[dict[str, Any]]:
        """Flush: a tool block cut by the budget becomes raw content; the held tail is emitted trimmed."""

        if self.stop_hit:
            return []
        if self.phase == "tool":
            self.truncated_tool_call = True
            self.phase = "content"
            return self._text("<tool_call>" + self.decode(self.tool_ids)) + self._flush()
        return self._flush()

    def message(self) -> dict[str, Any]:
        """The non-streamed ``choices[0].message`` (content null when empty, per the OpenAI tool-call shape)."""

        content = "".join(self.content)
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            message["tool_calls"] = [
                {key: value for key, value in call.items() if key != "index"} for call in self.calls
            ]
        return message

    def _enter(self, phase: str) -> list[dict[str, Any]]:
        deltas = self._flush()
        self.phase = phase
        return deltas

    def _flush(self) -> list[dict[str, Any]]:
        text = self.decode(self.pending_ids) if self.pending_ids else ""
        self.pending_ids = []
        buffer, self.held_text = self.held_text + text, ""
        if not self.phase_text_started:
            return []
        self.phase_text_started = False
        stop = self._stop_index(buffer)
        return self._emit(buffer[: len(buffer) if stop is None else stop].rstrip())

    def _stop_index(self, buffer: str) -> int | None:
        """Where the first stop string starts in ``buffer`` (content phase only); sets ``stop_hit``."""

        if self.phase != "content" or not self.stop_strings:
            return None
        hits = [buffer.find(value) for value in self.stop_strings if value in buffer]
        if not hits:
            return None
        self.stop_hit = True
        return min(hits)

    def _text(self, text: str) -> list[dict[str, Any]]:
        buffer = self.held_text + text
        if not self.phase_text_started:
            buffer = buffer.lstrip()
            if not buffer:
                self.held_text = ""
                return []
            self.phase_text_started = True
        hold = 0
        if self.phase == "content" and self.stop_strings:
            stop = self._stop_index(buffer)
            if stop is not None:
                self.held_text = ""
                return self._emit(buffer[:stop].rstrip())
            hold = self.stop_hold
        # Emit up to the hold-back window, never ending in whitespace (the template trims it; a stop match
        # right after would otherwise leave a stray space).
        cut = len(buffer[: max(0, len(buffer) - hold)].rstrip())
        self.held_text = buffer[cut:]
        return self._emit(buffer[:cut])

    def _emit(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        if self.phase == "reasoning":
            self.reasoning.append(text)
            return [{"reasoning_content": text}]
        self.content.append(text)
        return [{"content": text}]
