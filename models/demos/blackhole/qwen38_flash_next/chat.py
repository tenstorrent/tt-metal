# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pinned text-only chat formatting and safe local tools for Qwen3.8.

This module deliberately owns no model or device state.  It renders the exact
released ``chat_template.jinja``, separates generated reasoning from visible
content, parses the checkpoint's XML-like function-call syntax, and executes a
small allowlisted local tool.  The TTNN demo supplies token IDs to and from this
boundary; there is no alternate hand-written prompt template.

Vision items are rejected even though the multimodal checkpoint template can
render them.  The required bring-up is text-only and must not accidentally
claim support for a vision encoder which is not resident on the four-card mesh.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

CHECKPOINT_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
VOCAB_SIZE = 248_320
TOKENIZER_SIZE = 248_077
MAX_CONTEXT = 262_144
IM_START_ID = 248_045
IM_END_ID = 248_046
END_OF_TEXT_ID = 248_044
EOS_TOKEN_IDS = (IM_END_ID, END_OF_TEXT_ID)

PINNED_TOKENIZER_ARTIFACTS = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
}
REASONING_EFFORTS = ("xhigh", "medium", "low")

_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_.-]{0,127})>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_.-]{0,127})>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_TERMINATORS = ("<|im_end|>", "<|endoftext|>")


class Qwen38ChatFormatError(ValueError):
    """A message or generated response violates the pinned template contract."""


class Qwen38ToolExecutionError(RuntimeError):
    """A requested local tool is absent, malformed, or outside its safe domain."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise Qwen38ChatFormatError(f"required tokenizer artifact is unavailable: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise Qwen38ChatFormatError(f"tokenizer artifact must be a non-symlink regular file: {path}")


def _normal_reasoning_effort(value: str) -> Literal["xhigh", "medium", "low"]:
    if value not in REASONING_EFFORTS:
        raise Qwen38ChatFormatError(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {value!r}")
    return value  # type: ignore[return-value]


def _text_content(content: Any, *, label: str) -> str | list[dict[str, str]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise Qwen38ChatFormatError(f"{label} content must be text or a sequence of text items")
    result: list[dict[str, str]] = []
    for index, item in enumerate(content):
        if not isinstance(item, Mapping):
            raise Qwen38ChatFormatError(f"{label} content item {index} must be a mapping")
        if any(key in item for key in ("image", "image_url", "video")) or item.get("type") in {
            "image",
            "image_url",
            "video",
        }:
            raise Qwen38ChatFormatError("vision content is unsupported by the text-only TTNN demo")
        if item.get("type", "text") != "text" or not isinstance(item.get("text"), str):
            raise Qwen38ChatFormatError(f"{label} content item {index} is not an exact text item")
        result.append({"type": "text", "text": item["text"]})
    return result


def _validate_tool_schema(tool: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise Qwen38ChatFormatError(f"tool {index} must be a mapping")
    try:
        serialized = json.dumps(tool, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise Qwen38ChatFormatError(f"tool {index} is not finite JSON") from error
    function = result.get("function") if result.get("type") == "function" else None
    if not isinstance(function, dict) or not _FUNCTION_NAME.fullmatch(str(function.get("name", ""))):
        raise Qwen38ChatFormatError(f"tool {index} must contain a valid function name")
    if not isinstance(function.get("description"), str) or not isinstance(function.get("parameters"), dict):
        raise Qwen38ChatFormatError(f"tool {index} requires a description and JSON-schema parameters")
    return result


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise Qwen38ChatFormatError("chat history must be a nonempty message sequence")
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise Qwen38ChatFormatError(f"message {index} must be a mapping")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise Qwen38ChatFormatError(f"message {index} has unsupported role {role!r}")
        if role == "system" and index != 0:
            raise Qwen38ChatFormatError("the system message must be first")
        copied: dict[str, Any] = {
            "role": role,
            "content": _text_content(message.get("content", ""), label=f"message {index}"),
        }
        if role == "assistant":
            reasoning = message.get("reasoning_content", "")
            if not isinstance(reasoning, str):
                raise Qwen38ChatFormatError(f"assistant message {index} reasoning_content must be text")
            copied["reasoning_content"] = reasoning
            calls = message.get("tool_calls", ())
            if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence):
                raise Qwen38ChatFormatError(f"assistant message {index} tool_calls must be a sequence")
            normalized_calls = []
            for call_index, call in enumerate(calls):
                if not isinstance(call, Mapping):
                    raise Qwen38ChatFormatError(f"assistant message {index} tool call {call_index} must be a mapping")
                function = call.get("function", call)
                if not isinstance(function, Mapping):
                    raise Qwen38ChatFormatError(f"assistant message {index} tool call {call_index} has no function")
                name = function.get("name")
                arguments = function.get("arguments", {})
                if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
                    raise Qwen38ChatFormatError(f"assistant message {index} tool call {call_index} has an invalid name")
                if not isinstance(arguments, Mapping):
                    raise Qwen38ChatFormatError(
                        f"assistant message {index} tool call {call_index} arguments must be a mapping"
                    )
                normalized_calls.append({"type": "function", "function": {"name": name, "arguments": dict(arguments)}})
            if normalized_calls:
                copied["tool_calls"] = normalized_calls
        result.append(copied)
    if not any(message["role"] == "user" for message in result):
        raise Qwen38ChatFormatError("chat history requires a user query")
    return result


@dataclass(frozen=True)
class Qwen38RenderedPrompt:
    text: str
    input_ids: torch.Tensor
    enable_thinking: bool
    preserve_thinking: bool
    reasoning_effort: Literal["xhigh", "medium", "low"]
    template_sha256: str


class Qwen38OfficialChatTemplate:
    """Hash-verified local tokenizer and the released Jinja chat template."""

    def __init__(self, checkpoint_root: str | os.PathLike[str]) -> None:
        root = Path(checkpoint_root).resolve(strict=True)
        if not root.is_dir():
            raise Qwen38ChatFormatError(f"checkpoint root is not a directory: {root}")
        for name, expected in PINNED_TOKENIZER_ARTIFACTS.items():
            path = root / name
            _regular_file(path)
            actual = _sha256(path)
            if actual != expected:
                raise Qwen38ChatFormatError(
                    f"pinned tokenizer artifact {name} has SHA-256 {actual}, expected {expected}"
                )

        # Loading a tokenizer is a host formatting operation, not model
        # inference.  Network and remote-code execution are both disabled.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        # The model reserves 243 padded output rows above the released text
        # tokenizer.  Those rows are valid LM-head indices but are never prompt
        # tokens; do not conflate the two sizes.
        if len(tokenizer) != TOKENIZER_SIZE or tokenizer.vocab_size != END_OF_TEXT_ID:
            raise Qwen38ChatFormatError(
                f"tokenizer sizes {(tokenizer.vocab_size, len(tokenizer))} differ from "
                f"the pinned {(END_OF_TEXT_ID, TOKENIZER_SIZE)}"
            )
        if tokenizer.convert_tokens_to_ids("<|im_start|>") != IM_START_ID:
            raise Qwen38ChatFormatError("tokenizer <|im_start|> ID differs from the pinned checkpoint")
        if tokenizer.convert_tokens_to_ids("<|im_end|>") != IM_END_ID:
            raise Qwen38ChatFormatError("tokenizer <|im_end|> ID differs from the pinned checkpoint")
        if tokenizer.eos_token_id != IM_END_ID or tokenizer.pad_token_id != END_OF_TEXT_ID:
            raise Qwen38ChatFormatError("tokenizer EOS/pad IDs differ from the pinned chat contract")
        expected_template = (root / "chat_template.jinja").read_text(encoding="utf-8")
        if tokenizer.chat_template != expected_template:
            raise Qwen38ChatFormatError("loaded tokenizer did not select the pinned standalone chat template")
        self.checkpoint_root = root
        self.tokenizer = tokenizer

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        enable_thinking: bool = True,
        preserve_thinking: bool = True,
        reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh",
    ) -> Qwen38RenderedPrompt:
        if type(enable_thinking) is not bool or type(preserve_thinking) is not bool:
            raise Qwen38ChatFormatError("enable_thinking and preserve_thinking must be booleans")
        effort = _normal_reasoning_effort(reasoning_effort)
        normalized_messages = _validate_messages(messages)
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise Qwen38ChatFormatError("tools must be a sequence")
        normalized_tools = [_validate_tool_schema(tool, index=index) for index, tool in enumerate(tools)]
        rendered = self.tokenizer.apply_chat_template(
            normalized_messages,
            tools=normalized_tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            reasoning_effort=effort,
        )
        if not isinstance(rendered, str) or not rendered.endswith(
            "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        ):
            raise Qwen38ChatFormatError("official template generation suffix drifted")
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        if encoded.device.type != "cpu" or encoded.dtype != torch.int64 or tuple(encoded.shape[:1]) != (1,):
            raise Qwen38ChatFormatError("official chat tokenizer did not return CPU int64 global-B1 IDs")
        if encoded.shape[1] <= 0 or encoded.shape[1] > MAX_CONTEXT:
            raise Qwen38ChatFormatError(f"rendered chat length {encoded.shape[1]} is outside [1,{MAX_CONTEXT}]")
        if int(encoded.min()) < 0 or int(encoded.max()) >= VOCAB_SIZE:
            raise Qwen38ChatFormatError("rendered chat contains an out-of-vocabulary token")
        return Qwen38RenderedPrompt(
            text=rendered,
            input_ids=encoded.contiguous(),
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            reasoning_effort=effort,
            template_sha256=PINNED_TOKENIZER_ARTIFACTS["chat_template.jinja"],
        )


@dataclass(frozen=True)
class Qwen38ToolCall:
    name: str
    arguments: Mapping[str, Any]

    def as_template_value(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "arguments": dict(self.arguments)}}


@dataclass(frozen=True)
class Qwen38AssistantCompletion:
    reasoning_content: str
    content: str
    tool_calls: tuple[Qwen38ToolCall, ...]

    def as_template_message(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "reasoning_content": self.reasoning_content,
        }
        if self.tool_calls:
            value["tool_calls"] = [call.as_template_value() for call in self.tool_calls]
        return value


def _strip_one_terminator(text: str) -> str:
    found = [(text.find(token), token) for token in _TERMINATORS if token in text]
    if not found:
        return text
    index, token = min(found)
    suffix = text[index + len(token) :]
    if suffix.strip():
        raise Qwen38ChatFormatError("assistant output contains text after its terminal token")
    prefix = text[:index]
    if any(other in prefix for other in _TERMINATORS):
        raise Qwen38ChatFormatError("assistant output contains multiple terminal tokens")
    return prefix


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _parse_argument(value: str, declared: Any = None) -> Any:
    """A ``<parameter>`` body typed by its schema: the template renders a string argument raw and every other type
    with ``tojson``, so a parameter declared ``string`` stays verbatim and any other (or an undeclared one) is parsed
    when it is JSON, else kept as text.  NaN and Infinity are not JSON and stay text."""

    stripped = value.strip()
    types = {declared} if isinstance(declared, str) else set(declared) if isinstance(declared, list) else set()
    if "string" in types and not (stripped == "null" and "null" in types):
        return stripped
    if not stripped:
        return ""
    try:
        return json.loads(stripped, parse_constant=_reject_constant)
    except ValueError:
        return stripped


def _parameter_types(tools: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Function name -> parameter name -> the declared JSON-schema ``type`` (``parameters.properties``)."""

    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        result[str(function.get("name"))] = {
            name: schema.get("type")
            for name, schema in (properties.items() if isinstance(properties, Mapping) else ())
            if isinstance(schema, Mapping)
        }
    return result


def parse_tool_calls(text: str, tools: Sequence[Mapping[str, Any]] = ()) -> tuple[str, tuple[Qwen38ToolCall, ...]]:
    """The visible text before the first ``<tool_call>`` and the calls after it, their arguments typed by ``tools``
    (``parse_assistant_completion`` on the content part; the server's assembler on one block, whose parameter
    text may quote any tag)."""

    marker = text.find("<tool_call>")
    if marker < 0:
        if "</tool_call>" in text or "<function=" in text or "<parameter=" in text:
            raise Qwen38ChatFormatError("assistant output contains an incomplete tool-call tag")
        return text.strip(), ()
    visible = text[:marker].strip()
    tail = text[marker:]
    declared = _parameter_types(tools)
    calls: list[Qwen38ToolCall] = []
    cursor = 0
    while cursor < len(tail):
        whitespace = re.match(r"\s*", tail[cursor:])
        assert whitespace is not None
        cursor += whitespace.end()
        if cursor == len(tail):
            break
        match = _TOOL_CALL.match(tail, cursor)
        if match is None:
            raise Qwen38ChatFormatError("assistant tool-call tail is malformed or has a suffix")
        name, body = match.groups()
        arguments: dict[str, Any] = {}
        body_cursor = 0
        while body_cursor < len(body):
            whitespace = re.match(r"\s*", body[body_cursor:])
            assert whitespace is not None
            body_cursor += whitespace.end()
            if body_cursor == len(body):
                break
            parameter = _PARAMETER.match(body, body_cursor)
            if parameter is None:
                raise Qwen38ChatFormatError(f"tool {name!r} contains malformed parameter syntax")
            parameter_name, value = parameter.groups()
            if parameter_name in arguments:
                raise Qwen38ChatFormatError(f"tool {name!r} repeats parameter {parameter_name!r}")
            arguments[parameter_name] = _parse_argument(value, declared.get(name, {}).get(parameter_name))
            body_cursor = parameter.end()
        calls.append(Qwen38ToolCall(name=name, arguments=arguments))
        cursor = match.end()
    if not calls:
        raise Qwen38ChatFormatError("assistant output opened an empty tool-call tail")
    return visible, tuple(calls)


def parse_assistant_completion(
    text: str, *, enable_thinking: bool, tools: Sequence[Mapping[str, Any]] = ()
) -> Qwen38AssistantCompletion:
    """Parse only the newly generated assistant suffix.

    For thinking mode the rendered prompt already ends in ``<think>\n``; the
    generated suffix therefore begins with reasoning and must close that tag.
    For non-thinking mode the prompt already contains the complete empty think
    block and the generated suffix begins directly with visible content.
    ``tools`` (the request's schemas) type the tool-call arguments; without
    them every argument that is JSON is parsed.
    """

    if not isinstance(text, str):
        raise TypeError("assistant completion must be text")
    if type(enable_thinking) is not bool:
        raise TypeError("enable_thinking must be a boolean")
    generated = _strip_one_terminator(text)
    if enable_thinking:
        if generated.startswith("<think>\n"):
            generated = generated[len("<think>\n") :]
        if generated.count("</think>") != 1:
            raise Qwen38ChatFormatError("thinking output must contain exactly one </think> delimiter")
        reasoning, visible = generated.split("</think>", 1)
        reasoning = reasoning.strip()
        visible = visible.strip()
    else:
        if "<think>" in generated or "</think>" in generated:
            raise Qwen38ChatFormatError("non-thinking output must not emit another think block")
        reasoning = ""
        visible = generated.strip()
    content, calls = parse_tool_calls(visible, tools)
    return Qwen38AssistantCompletion(
        reasoning_content=reasoning,
        content=content,
        tool_calls=calls,
    )


SAFE_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add_integers",
        "description": "Add two bounded integers locally without network or filesystem access.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}


class Qwen38SafeLocalTools:
    """The demo's closed, side-effect-free local tool allowlist."""

    @property
    def schemas(self) -> tuple[dict[str, Any], ...]:
        return (json.loads(json.dumps(SAFE_ADD_TOOL)),)

    def execute(self, call: Qwen38ToolCall) -> str:
        if type(call) is not Qwen38ToolCall:
            raise Qwen38ToolExecutionError("safe tool runner requires an exact Qwen38ToolCall")
        if call.name != "add_integers":
            raise Qwen38ToolExecutionError(f"tool {call.name!r} is not in the local allowlist")
        if set(call.arguments) != {"a", "b"}:
            raise Qwen38ToolExecutionError("add_integers requires exactly parameters a and b")
        values = []
        for name in ("a", "b"):
            value = call.arguments[name]
            if isinstance(value, bool) or not isinstance(value, int) or not -(10**9) <= value <= 10**9:
                raise Qwen38ToolExecutionError(f"add_integers parameter {name} must be a bounded integer")
            values.append(value)
        return json.dumps({"a": values[0], "b": values[1], "sum": sum(values)}, separators=(",", ":"))


@dataclass
class Qwen38ChatConversation:
    """Inspectable multi-turn history in the exact template message schema."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.messages:
            self.messages = _validate_messages(self.messages)

    def add_system(self, content: str) -> None:
        if self.messages:
            raise Qwen38ChatFormatError("a system message can only be added first")
        self.messages.append({"role": "system", "content": _text_content(content, label="system")})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": _text_content(content, label="user")})

    def add_assistant(self, completion: Qwen38AssistantCompletion) -> None:
        if type(completion) is not Qwen38AssistantCompletion:
            raise Qwen38ChatFormatError("assistant history requires a parsed Qwen38 completion")
        self.messages.append(completion.as_template_message())

    def execute_tool_round(self, tools: Qwen38SafeLocalTools) -> tuple[str, ...]:
        if not self.messages or self.messages[-1].get("role") != "assistant":
            raise Qwen38ToolExecutionError("the last history message is not an assistant tool call")
        raw_calls = self.messages[-1].get("tool_calls", ())
        if not raw_calls:
            raise Qwen38ToolExecutionError("the last assistant message contains no tool call")
        responses = []
        for raw in raw_calls:
            function = raw["function"]
            call = Qwen38ToolCall(function["name"], function["arguments"])
            response = tools.execute(call)
            self.messages.append({"role": "tool", "content": response})
            responses.append(response)
        return tuple(responses)

    def render(
        self,
        template: Qwen38OfficialChatTemplate,
        *,
        tools: Qwen38SafeLocalTools | None = None,
        enable_thinking: bool = True,
        preserve_thinking: bool = True,
        reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh",
    ) -> Qwen38RenderedPrompt:
        return template.render(
            self.messages,
            tools=() if tools is None else tools.schemas,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            reasoning_effort=reasoning_effort,
        )


__all__ = [
    "CHECKPOINT_REVISION",
    "EOS_TOKEN_IDS",
    "PINNED_TOKENIZER_ARTIFACTS",
    "REASONING_EFFORTS",
    "SAFE_ADD_TOOL",
    "Qwen38AssistantCompletion",
    "Qwen38ChatConversation",
    "Qwen38ChatFormatError",
    "Qwen38OfficialChatTemplate",
    "Qwen38RenderedPrompt",
    "Qwen38SafeLocalTools",
    "Qwen38ToolCall",
    "Qwen38ToolExecutionError",
    "parse_assistant_completion",
    "parse_tool_calls",
]
