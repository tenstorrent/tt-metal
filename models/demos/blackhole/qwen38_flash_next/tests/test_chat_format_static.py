# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only gates for the pinned Qwen3.8 chat and local-tool boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.chat import (
    EOS_TOKEN_IDS,
    PINNED_TOKENIZER_ARTIFACTS,
    Qwen38AssistantCompletion,
    Qwen38ChatConversation,
    Qwen38ChatFormatError,
    Qwen38OfficialChatTemplate,
    Qwen38SafeLocalTools,
    Qwen38ToolCall,
    Qwen38ToolExecutionError,
    parse_assistant_completion,
)

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))


@pytest.fixture(scope="module")
def template() -> Qwen38OfficialChatTemplate:
    return Qwen38OfficialChatTemplate(CHECKPOINT)


def test_exact_artifacts_and_official_thinking_suffix(template: Qwen38OfficialChatTemplate) -> None:
    conversation = Qwen38ChatConversation()
    conversation.add_system("Answer precisely.")
    conversation.add_user("What is two plus three?")

    xhigh = conversation.render(template, enable_thinking=True, reasoning_effort="xhigh")
    assert xhigh.template_sha256 == PINNED_TOKENIZER_ARTIFACTS["chat_template.jinja"]
    assert xhigh.text.endswith("<|im_start|>assistant\n<think>\n")
    assert "Reasoning effort is set to xhigh." in xhigh.text
    assert tuple(xhigh.input_ids.shape[:1]) == (1,)
    assert xhigh.input_ids.shape[1] > 0

    medium = conversation.render(template, enable_thinking=True, reasoning_effort="medium")
    assert "Reasoning effort is set to" not in medium.text
    assert medium.text.endswith("<think>\n")


def test_nonthinking_suffix_and_effort_validation(expect_error, template: Qwen38OfficialChatTemplate) -> None:
    conversation = Qwen38ChatConversation()
    conversation.add_user("Reply without thinking.")
    rendered = conversation.render(template, enable_thinking=False, reasoning_effort="low")
    assert rendered.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Reasoning effort is set" not in rendered.text
    with expect_error(Qwen38ChatFormatError, match="reasoning_effort"):  # allow-pytest.raises: pure host gate
        conversation.render(template, reasoning_effort="high")  # type: ignore[arg-type]


def test_text_only_boundary_rejects_vision(expect_error, template: Qwen38OfficialChatTemplate) -> None:
    messages = [{"role": "user", "content": [{"type": "image", "image_url": "local.png"}]}]
    with expect_error(Qwen38ChatFormatError, match="vision content is unsupported"):  # allow-pytest.raises
        template.render(messages)


def test_thinking_completion_and_exact_tool_call_round_trip(template: Qwen38OfficialChatTemplate) -> None:
    generated = """Check the arithmetic.
</think>

<tool_call>
<function=add_integers>
<parameter=a>
2
</parameter>
<parameter=b>
3
</parameter>
</function>
</tool_call><|im_end|>"""
    completion = parse_assistant_completion(generated, enable_thinking=True)
    assert completion.reasoning_content == "Check the arithmetic."
    assert completion.content == ""
    assert completion.tool_calls == (Qwen38ToolCall("add_integers", {"a": 2, "b": 3}),)

    tools = Qwen38SafeLocalTools()
    conversation = Qwen38ChatConversation()
    conversation.add_user("Use the tool to add 2 and 3.")
    conversation.add_assistant(completion)
    assert conversation.execute_tool_round(tools) == ('{"a":2,"b":3,"sum":5}',)

    rendered = conversation.render(
        template,
        tools=tools,
        enable_thinking=True,
        preserve_thinking=True,
        reasoning_effort="low",
    )
    assert "<function=add_integers>" in rendered.text
    assert '<tool_response>\n{"a":2,"b":3,"sum":5}\n</tool_response>' in rendered.text
    assert "<think>\nCheck the arithmetic.\n</think>" in rendered.text


def test_preserve_thinking_false_uses_official_template(template: Qwen38OfficialChatTemplate) -> None:
    conversation = Qwen38ChatConversation(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "done", "reasoning_content": "private old reasoning"},
            {"role": "user", "content": "second"},
        ]
    )
    hidden = conversation.render(template, preserve_thinking=False)
    preserved = conversation.render(template, preserve_thinking=True)
    assert "private old reasoning" not in hidden.text
    assert "private old reasoning" in preserved.text


@pytest.mark.parametrize(
    "generated,pattern",
    [
        ("missing close", "exactly one"),
        ("x</think>y</think>", "exactly one"),
        ("x</think><tool_call>broken", "malformed"),
        ("x</think><function=add_integers></function>", "incomplete"),
        ("x</think>answer<|im_end|>suffix", "after its terminal"),
    ],
)
def test_malformed_generated_protocol_fails_closed(expect_error, generated: str, pattern: str) -> None:
    with expect_error(Qwen38ChatFormatError, match=pattern):  # allow-pytest.raises: parser rejection gate
        parse_assistant_completion(generated, enable_thinking=True)


def test_duplicate_parameters_and_tool_suffix_fail_closed(expect_error) -> None:
    duplicate = """r</think><tool_call><function=add_integers>
<parameter=a>1</parameter><parameter=a>2</parameter><parameter=b>3</parameter>
</function></tool_call>"""
    with expect_error(Qwen38ChatFormatError, match="repeats parameter"):  # allow-pytest.raises
        parse_assistant_completion(duplicate, enable_thinking=True)

    suffix = """r</think><tool_call><function=add_integers>
<parameter=a>1</parameter><parameter=b>2</parameter>
</function></tool_call>not allowed"""
    with expect_error(Qwen38ChatFormatError, match="suffix"):  # allow-pytest.raises
        parse_assistant_completion(suffix, enable_thinking=True)


def test_tool_arguments_follow_the_schema_types() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "count": {"type": "integer"}, "tags": {"type": "array"}},
                },
            },
        }
    ]
    generated = (
        "<tool_call><function=lookup><parameter=id>42</parameter><parameter=count>3</parameter>"
        "<parameter=tags>[1]</parameter><parameter=other>true</parameter><parameter=weight>NaN</parameter>"
        "</function></tool_call>"
    )
    typed = parse_assistant_completion(generated, enable_thinking=False, tools=tools).tool_calls[0].arguments
    assert typed == {"id": "42", "count": 3, "tags": [1], "other": True, "weight": "NaN"}
    guessed = parse_assistant_completion(generated, enable_thinking=False).tool_calls[0].arguments
    assert guessed == {"id": 42, "count": 3, "tags": [1], "other": True, "weight": "NaN"}


def test_nonthinking_parser_has_no_reasoning(expect_error) -> None:
    completion = parse_assistant_completion("visible answer<|im_end|>", enable_thinking=False)
    assert completion == Qwen38AssistantCompletion("", "visible answer", ())
    with expect_error(Qwen38ChatFormatError, match="must not emit"):  # allow-pytest.raises
        parse_assistant_completion("<think>x</think>y", enable_thinking=False)


def test_safe_tool_is_closed_and_bounded(expect_error) -> None:
    tools = Qwen38SafeLocalTools()
    assert tuple(schema["function"]["name"] for schema in tools.schemas) == ("add_integers",)
    assert tools.execute(Qwen38ToolCall("add_integers", {"a": -4, "b": 9})) == '{"a":-4,"b":9,"sum":5}'
    with expect_error(Qwen38ToolExecutionError, match="not in the local allowlist"):  # allow-pytest.raises
        tools.execute(Qwen38ToolCall("shell", {"command": "id"}))
    with expect_error(Qwen38ToolExecutionError, match="exactly parameters"):  # allow-pytest.raises
        tools.execute(Qwen38ToolCall("add_integers", {"a": 1, "b": 2, "path": "/tmp"}))
    with expect_error(Qwen38ToolExecutionError, match="bounded integer"):  # allow-pytest.raises
        tools.execute(Qwen38ToolCall("add_integers", {"a": True, "b": 2}))


def test_pinned_eos_set_covers_chat_end_and_endoftext() -> None:
    assert EOS_TOKEN_IDS == (248_046, 248_044)
