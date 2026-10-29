# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fakes shared by the chat, served-request and vision no-device tests."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import jinja2
import pytest

from models.demos.blackhole.qwen38_flash_next.chat import IM_END_ID, IM_START_ID
from models.demos.blackhole.qwen38_flash_next.mrope import IMAGE_TOKEN_ID, VISION_END_TOKEN_ID, VISION_START_TOKEN_ID
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import (
    THINK_END_ID,
    THINK_START_ID,
    TOOL_CALL_END_ID,
    TOOL_CALL_START_ID,
    Qwen38ChatRequestRejected,
)

CHAT_SPECIAL_TOKENS = {
    "<|im_start|>": IM_START_ID,
    "<|im_end|>": IM_END_ID,
    "<think>": THINK_START_ID,
    "</think>": THINK_END_ID,
    "<tool_call>": TOOL_CALL_START_ID,
    "</tool_call>": TOOL_CALL_END_ID,
    "<|image_pad|>": IMAGE_TOKEN_ID,
    "<|vision_start|>": VISION_START_TOKEN_ID,
    "<|vision_end|>": VISION_END_TOKEN_ID,
}
CHAT_SPECIAL_TEXT = {value: key for key, value in CHAT_SPECIAL_TOKENS.items()}


class ChatTemplateFakeTokenizer:
    """A template of the checkpoint's shape (turn markers, think block, XML tool blocks, tool responses as user turns,
    an image item as its vision pad) and a tokenizer that maps special tags to their ids and every other character to
    its code point. ``calls`` records each render's arguments."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        preserve_thinking,
        reasoning_effort,
    ) -> str:
        assert tokenize is False and preserve_thinking is True
        if reasoning_effort not in ("xhigh", "medium", "low"):
            raise jinja2.exceptions.TemplateError(f"Unexpected reasoning effort {reasoning_effort}.")
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        text = lambda content: (  # noqa: E731
            content
            if isinstance(content, str)
            else "".join(
                item["text"] if item["type"] == "text" else "<|vision_start|><|image_pad|><|vision_end|>"
                for item in content
            )
        ).strip()
        parts = []
        if tools:
            system = text(messages[0]["content"]) if messages[0]["role"] == "system" else ""
            parts.append(
                "<|im_start|>system\n# Tools\n"
                + "\n".join(json.dumps(tool) for tool in tools)
                + ("\n\n" + system if system else "")
                + "<|im_end|>\n"
            )
        for index, message in enumerate(messages):
            content = text(message["content"])
            if message["role"] == "system":
                if index != 0:
                    raise jinja2.exceptions.TemplateError("System message must be at the beginning.")
                if not tools and content:
                    parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif message["role"] == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif message["role"] == "assistant":
                reasoning = message.get("reasoning_content", "").strip()
                turn = f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}"
                for index, call in enumerate(message.get("tool_calls", ())):
                    function = call["function"]
                    parameters = "".join(
                        f"<parameter={name}>\n{value if isinstance(value, str) else json.dumps(value)}\n</parameter>\n"
                        for name, value in function["arguments"].items()
                    )
                    # The checkpoint's rule: a blank line before the first call after content, one newline otherwise.
                    separator = ("\n\n" if content else "") if index == 0 else "\n"
                    turn += (
                        f"{separator}<tool_call>\n<function={function['name']}>\n{parameters}</function>\n</tool_call>"
                    )
                parts.append(turn + "<|im_end|>\n")
            elif message["role"] == "tool":
                parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
            else:
                raise jinja2.exceptions.TemplateError("Unexpected message role.")
        if not any(message["role"] == "user" for message in messages):
            raise jinja2.exceptions.TemplateError("No user query found in messages.")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
        return "".join(parts)

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        ids = []
        position = 0
        while position < len(text):
            for tag, token_id in CHAT_SPECIAL_TOKENS.items():
                if text.startswith(tag, position):
                    ids.append(token_id)
                    position += len(tag)
                    break
            else:
                ids.append(ord(text[position]))
                position += 1
        return SimpleNamespace(input_ids=ids)


def chat_request_rejection(function, *args, **kwargs) -> Qwen38ChatRequestRejected:
    """The ``Qwen38ChatRequestRejected`` that ``function(*args, **kwargs)`` raises, for its ``param`` and ``code``."""

    with pytest.raises(Qwen38ChatRequestRejected) as info:  # allow-pytest.raises: inspect the captured exception object
        function(*args, **kwargs)
    return info.value


def skip_without_git_checkout(root) -> None:
    if subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True).returncode:
        pytest.skip(f"{root} is not a git checkout")
