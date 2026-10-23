# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chat protocol module without a device: request normalisation, reply assembly (fake tokenizer), and, with the
pinned checkpoint (4x p150), the template-equality matrix: the server's prompt ids must equal
``tokenizer.apply_chat_template`` on the raw request bitwise for every message shape x tools x thinking case, and the
prefix a follow-up turn's reference render shares with the served turn."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest

from models.demos.blackhole.qwen38_flash_next.chat import IM_END_ID, IM_START_ID, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import (
    THINK_END_ID,
    THINK_START_ID,
    TOOL_CALL_END_ID,
    TOOL_CALL_START_ID,
    Qwen38ChatRequestRejected,
    Qwen38ReplyAssembler,
)

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
SPECIAL_TOKENS = {
    "<|im_start|>": IM_START_ID,
    "<|im_end|>": IM_END_ID,
    "<think>": THINK_START_ID,
    "</think>": THINK_END_ID,
    "<tool_call>": TOOL_CALL_START_ID,
    "</tool_call>": TOOL_CALL_END_ID,
}
SPECIAL_TEXT = {value: key for key, value in SPECIAL_TOKENS.items()}


class FakeTokenizer:
    """A template of the checkpoint's shape (turn markers, think block, XML tool blocks, tool responses as user
    turns) and a tokenizer that maps special tags to their ids and every other character to its code point."""

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
                "preserve_thinking": preserve_thinking,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        text = lambda content: (  # noqa: E731
            content if isinstance(content, str) else "".join(item["text"] for item in content)
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
                turn = f"<|im_start|>assistant\n<think>\n{message.get('reasoning_content', '').strip()}\n</think>\n\n{content}"
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
            for tag, token_id in SPECIAL_TOKENS.items():
                if text.startswith(tag, position):
                    ids.append(token_id)
                    position += len(tag)
                    break
            else:
                ids.append(ord(text[position]))
                position += 1
        return SimpleNamespace(input_ids=ids)

    @staticmethod
    def decode(ids, **_kw) -> str:
        return "".join(SPECIAL_TEXT.get(token, chr(token) if token < 0x110000 else "?") for token in ids)


FLAGS = {"enable_thinking": True, "reasoning_effort": "medium"}
ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add_integers",
        "description": "Add two integers.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
    },
}


# -- request side ---------------------------------------------------------------------------------


def test_normalize_messages_accepts_the_hermes_shapes_and_keeps_only_what_the_template_reads() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}], "name": "sam"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add_integers", "arguments": '{"b": 3, "a": 2}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "add_integers", "content": '{"sum":5}'},
        {"role": "assistant", "content": "done", "reasoning_content": None, "tool_calls": []},
    ]
    normalized = protocol.normalize_messages(messages)
    assert normalized == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "add_integers", "arguments": {"b": 3, "a": 2}}}],
        },
        {"role": "tool", "content": '{"sum":5}'},
        {"role": "assistant", "content": "done"},
    ]
    # Key order of string arguments survives (the template renders parameters in that order).
    assert list(normalized[2]["tool_calls"][0]["function"]["arguments"]) == ["b", "a"]
    assert protocol.normalize_messages(normalized) == normalized


@pytest.mark.parametrize(
    "messages, param, fragment",
    [
        ([], "messages", "nonempty"),
        ("text", "messages", "got str"),
        ([1], "messages[0]", "got int"),
        ([{"role": "robot", "content": "x"}], "messages[0].role", "got 'robot'"),
        ([{"role": "user", "content": "x"}, {"role": "system", "content": "y"}], "messages[1].role", "must be first"),
        ([{"role": "user", "content": 5}], "messages[0].content", "got int"),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}],
            "messages[0].content[0]",
            "text-only",
        ),
        ([{"role": "assistant", "content": "x", "reasoning_content": 1}], "messages[0].reasoning_content", "got int"),
        ([{"role": "assistant", "content": "x", "tool_calls": {}}], "messages[0].tool_calls", "got dict"),
        (
            [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{bad"}}]}],
            "messages[0].tool_calls[0].function.arguments",
            "not valid JSON",
        ),
        (
            [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}],
            "messages[0].tool_calls[0].function.arguments",
            "got list",
        ),
        (
            [{"role": "assistant", "tool_calls": [{"function": {"name": "bad name"}}]}],
            "messages[0].tool_calls[0].function.name",
            "got 'bad name'",
        ),
        ([{"role": "assistant", "tool_calls": [{"name": "f"}]}], "messages[0].tool_calls[0]", "'function'"),
    ],
)
def test_normalize_messages_rejects_with_param_and_actual_vs_expected(messages, param, fragment) -> None:
    with pytest.raises(Qwen38ChatRequestRejected) as info:  # allow-pytest.raises: inspect the captured exception object
        protocol.normalize_messages(messages)
    assert info.value.param == param and info.value.code == "bad_request" and fragment in str(info.value)


def test_validate_tools_keeps_key_order_and_rejects_bad_shapes() -> None:
    tool = {"function": {"parameters": {"z": 1, "a": 2}, "name": "f", "description": "d"}, "type": "function"}
    [validated] = protocol.validate_tools([tool])
    assert json.dumps(validated) == json.dumps(tool)  # not re-sorted
    assert list(validated) == ["function", "type"] and list(validated["function"]["parameters"]) == ["z", "a"]
    assert protocol.validate_tools(None) == [] and protocol.validate_tools([]) == []
    assert protocol.validate_tools([{"type": "function", "function": {"name": "g"}}]) == [
        {"type": "function", "function": {"name": "g"}}
    ]
    for bad, param in (
        ({}, "tools"),
        ([{"type": "retrieval"}], "tools[0]"),
        ([{"type": "function", "function": {"name": "1bad"}}], "tools[0].function.name"),
        ([{"type": "function", "function": {"name": "f", "description": 3}}], "tools[0].function.description"),
        ([{"type": "function", "function": {"name": "f", "parameters": []}}], "tools[0].function.parameters"),
        ([{"type": "function", "function": {"name": "f", "parameters": {"x": float("nan")}}}], "tools[0]"),
    ):
        with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
            Qwen38ChatRequestRejected
        ) as info:  # allow-pytest.raises: inspect the captured exception object
            protocol.validate_tools(bad)
        assert info.value.param == param


def test_stop_effort_and_thinking_budget_rules(expect_error) -> None:
    assert protocol.validate_stop(None) == () and protocol.validate_stop("END") == ("END",)
    assert protocol.validate_stop(["a", "b"]) == ("a", "b")
    for bad in ("", ["a", ""], ["1", "2", "3", "4", "5"], 3):
        with expect_error(Qwen38ChatRequestRejected):
            protocol.validate_stop(bad)
    assert [protocol.effort_level(value) for value in ("minimal", "low", "medium", "high", "xhigh")] == [
        "low",
        "low",
        "medium",
        "xhigh",
        "xhigh",
    ]
    with expect_error(Qwen38ChatRequestRejected, match="got 'max'"):
        protocol.effort_level("max")
    # low: 4096, medium: 16384, xhigh: unbounded; all leave 256 tokens (at least half of max_tokens) for the answer.
    assert protocol.THINKING_TOKEN_CAPS == {"low": 4096, "medium": 16_384, "xhigh": None}
    assert protocol.ANSWER_RESERVE_TOKENS == 256
    assert protocol.thinking_budget("low", 32_704) == 4096 and protocol.thinking_budget("medium", 32_704) == 16_384
    assert protocol.thinking_budget("xhigh", 32_704) == 32_447
    # A 12k prompt at 32k leaves 20,704: medium takes its cap, xhigh the room less the reserve.
    assert protocol.thinking_budget("medium", 32_704 - 12_000) == min(16_384, 32_704 - 12_000 - 256) == 16_384
    assert protocol.thinking_budget("xhigh", 32_704 - 12_000) == 20_447
    # Below the caps the reserve bounds every level: budget + the forced </think> + 256 answer tokens = max_tokens.
    assert protocol.thinking_budget("low", 4096) == 3839 == 4096 - 1 - protocol.ANSWER_RESERVE_TOKENS
    assert protocol.thinking_budget("medium", 4096) == 3839 and protocol.thinking_budget("xhigh", 4096) == 3839
    assert protocol.thinking_budget("medium", 2048) == 1791  # Hermes' max_tokens
    assert protocol.thinking_budget("xhigh", 300) == 150 and protocol.thinking_budget("low", 32) == 16
    assert protocol.thinking_budget("xhigh", 4096, requested=100) == 100
    assert protocol.thinking_budget("low", 4096, requested=0) == 0
    assert protocol.thinking_budget("low", 4096, requested=10_000) == 3839


def test_render_prompt_normalises_validates_and_encodes_with_the_generation_prompt(expect_error) -> None:
    tokenizer = FakeTokenizer()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": None}]
    ids = protocol.render_prompt(tokenizer, messages, [ADD_TOOL], **FLAGS)
    assert ids[:1] == [IM_START_ID] and ids[-2:] == [THINK_START_ID, ord("\n")]
    assert tokenizer.calls[-1]["tools"] == [ADD_TOOL] and tokenizer.calls[-1]["messages"][1]["content"] == ""
    ids_off = protocol.render_prompt(tokenizer, messages, [], enable_thinking=False, reasoning_effort="low")
    assert ids_off[-6:] == [THINK_START_ID, 10, 10, THINK_END_ID, 10, 10] and tokenizer.calls[-1]["tools"] is None
    with expect_error(Qwen38ChatRequestRejected, match="No user query"):
        protocol.render_prompt(
            tokenizer, [{"role": "system", "content": "s"}, {"role": "tool", "content": "r"}], [], **FLAGS
        )
    with expect_error(Qwen38ChatRequestRejected, match="Unexpected reasoning effort"):
        protocol.render_chat(
            tokenizer, protocol.normalize_messages(messages), [], enable_thinking=True, reasoning_effort="max"
        )


# -- reply side -----------------------------------------------------------------------------------


def _decoder(pieces: dict[int, str]):
    table = {**SPECIAL_TEXT, **pieces}
    return lambda ids: "".join(table[token] for token in ids)


def _run(assembler: Qwen38ReplyAssembler, tokens) -> list[dict]:
    deltas = []
    for token in tokens:
        deltas.extend(assembler.push(token))
    deltas.extend(assembler.finish())
    return deltas


def test_assembler_streams_content_and_trims_the_edges_the_template_trims() -> None:
    assembler = Qwen38ReplyAssembler(
        _decoder({1: "\n\n", 2: "Hello", 3: " world", 4: "\n", 5: "!"}), thinking_open=False
    )
    assert _run(assembler, [1, 2, 3, 4, 5, 4]) == [{"content": "Hello"}, {"content": " world"}, {"content": "\n!"}]
    assert assembler.message() == {"role": "assistant", "content": "Hello world\n!"}
    assert assembler.reasoning_tokens == 0 and assembler.tokens == 6 and assembler.calls == []


def test_assembler_splits_reasoning_from_content_at_the_think_end_token() -> None:
    decode = _decoder({1: "Let me", 2: " think.", 3: "\n\n", 4: "Answer", 5: "\n"})
    assembler = Qwen38ReplyAssembler(decode, thinking_open=True)
    deltas = _run(assembler, [1, 2, 5, THINK_END_ID, 3, 4, 5])
    assert deltas == [{"reasoning_content": "Let me"}, {"reasoning_content": " think."}, {"content": "Answer"}]
    assert assembler.reasoning_tokens == 3  # the reasoning tokens before </think>, whitespace included
    assert assembler.message() == {"role": "assistant", "content": "Answer", "reasoning_content": "Let me think."}
    # Non-thinking mode: the think tags are the answer's text (a quoted tag never opens a reasoning phase).
    assembler = Qwen38ReplyAssembler(decode, thinking_open=False)
    deltas = _run(assembler, [THINK_END_ID, 4, THINK_START_ID, 1, THINK_END_ID, 4])
    assert "".join(delta["content"] for delta in deltas) == "</think>Answer<think>Let me</think>Answer"
    assert assembler.reasoning == [] and assembler.reasoning_tokens == 0


def test_assembler_tag_ids_act_only_in_the_phase_that_expects_them() -> None:
    decode = _decoder({**TOOL_BLOCK, 30: "The tags are ", 31: " and ", 32: "; the reasoning ends here.", 33: " ok"})
    # (a) An answer that quotes the tags: no phase change, the literals stay in the content.
    assembler = Qwen38ReplyAssembler(decode, thinking_open=True)
    _run(assembler, [30, THINK_END_ID, 30, THINK_START_ID, 31, THINK_END_ID, 33])
    assert assembler.message() == {
        "role": "assistant",
        "content": "The tags are <think> and </think> ok",
        "reasoning_content": "The tags are",
    }
    # (b) Think tags inside a tool block are the argument's text; the call is parsed whole.
    assembler = Qwen38ReplyAssembler(decode, thinking_open=False, tools=[ADD_TOOL])
    _run(assembler, [TOOL_CALL_START_ID, 16, THINK_START_ID, THINK_END_ID, 19, TOOL_CALL_END_ID])
    assert assembler.parse_errors == [] and len(assembler.calls) == 1
    assert json.loads(assembler.calls[0]["function"]["arguments"]) == {"query": "<think></think>"}
    # (c) A tool block drafted inside the reasoning is reasoning: no call, nothing to execute, the answer follows.
    assembler = Qwen38ReplyAssembler(decode, thinking_open=True)
    deltas = _run(assembler, [14, TOOL_CALL_START_ID, 10, 11, 12, 13, TOOL_CALL_END_ID, 32, THINK_END_ID, 14])
    assert assembler.calls == [] and all("tool_calls" not in delta for delta in deltas)
    reasoning = "".join(assembler.reasoning)
    assert reasoning.startswith("I'll add them.<tool_call>\n<function=add_integers>") and reasoning.endswith(
        "</function>\n</tool_call>; the reasoning ends here."
    )
    assert assembler.message()["content"] == "I'll add them."
    # A <tool_call> opened inside a tool block is that block's text: the block fails to parse as one call and is
    # returned raw rather than restarted.
    assembler = Qwen38ReplyAssembler(decode, thinking_open=False)
    deltas = _run(assembler, [TOOL_CALL_START_ID, 10, TOOL_CALL_START_ID, 10, 11, 12, 13, TOOL_CALL_END_ID])
    assert assembler.calls == [] and len(assembler.parse_errors) == 1
    assert deltas[0]["content"].count("<tool_call>") == 2


TOOL_BLOCK = {
    10: "\n<function=add_integers>\n",
    11: "<parameter=a>\n2\n</parameter>\n",
    12: "<parameter=b>\n3\n</parameter>\n",
    13: "</function>\n",
    14: "I'll add them.",
    15: "\n\n",
    16: "\n<function=lookup>\n<parameter=query>\n",
    17: '{"x": [1, 2]}',
    18: "\nline two",
    19: "\n</parameter>\n</function>\n",
    20: "<parameter=a>\n",
    21: "Hel",
    22: "lo ST",
    23: "OP",
    24: " world",
}


def test_assembler_emits_a_completed_tool_call_as_one_delta_with_openai_fields() -> None:
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False)
    deltas = _run(assembler, [14, 15, TOOL_CALL_START_ID, 10, 11, 12, 13, TOOL_CALL_END_ID, 15])
    assert deltas[0] == {"content": "I'll add them."}
    [call] = deltas[1]["tool_calls"]
    assert (
        call["index"] == 0 and call["type"] == "function" and call["id"].startswith("call_") and len(call["id"]) == 17
    )
    assert call["function"] == {"name": "add_integers", "arguments": '{"a": 2, "b": 3}'}
    assert len(deltas) == 2
    message = assembler.message()
    assert message["content"] == "I'll add them." and message["tool_calls"] == [
        {k: v for k, v in call.items() if k != "index"}
    ]


def test_assembler_numbers_parallel_calls_keeps_multiline_and_json_parameters_and_nulls_empty_content() -> None:
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False)
    deltas = _run(
        assembler,
        [
            TOOL_CALL_START_ID,
            10,
            11,
            12,
            13,
            TOOL_CALL_END_ID,
            15,
            TOOL_CALL_START_ID,
            16,
            17,
            18,
            19,
            TOOL_CALL_END_ID,
        ],
    )
    assert [delta["tool_calls"][0]["index"] for delta in deltas] == [0, 1]
    second = json.loads(deltas[1]["tool_calls"][0]["function"]["arguments"])
    assert second == {"query": '{"x": [1, 2]}\nline two'}  # not JSON as a whole: kept verbatim, multiline
    assert assembler.message()["content"] is None and len(assembler.message()["tool_calls"]) == 2
    # A JSON value alone in a parameter is parsed (nested object).
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False)
    _run(assembler, [TOOL_CALL_START_ID, 16, 17, 19, TOOL_CALL_END_ID])
    assert json.loads(assembler.calls[0]["function"]["arguments"]) == {"query": {"x": [1, 2]}}


def test_assembler_falls_back_to_raw_content_for_malformed_and_truncated_tool_calls() -> None:
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False)
    deltas = _run(assembler, [TOOL_CALL_START_ID, 10, 20, 13, TOOL_CALL_END_ID])
    assert deltas == [{"content": "<tool_call>\n<function=add_integers>\n<parameter=a>\n</function>\n</tool_call>"}]
    assert assembler.calls == [] and len(assembler.parse_errors) == 1 and "malformed" in assembler.parse_errors[0]
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False)
    deltas = _run(assembler, [14, TOOL_CALL_START_ID, 10, 11])
    assert deltas == [
        {"content": "I'll add them."},
        {"content": "<tool_call>\n<function=add_integers>\n<parameter=a>\n2\n</parameter>"},
    ]
    assert assembler.truncated_tool_call is True and assembler.calls == []


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "note": {"type": ["string", "null"]},
                "lines": {"type": "integer"},
                "range": {"type": "object"},
                "flags": {"type": "array"},
                "exact": {"type": "boolean"},
            },
        },
    },
}
ARGUMENT_PIECES = {
    40: "\n<function=read_file>\n",
    41: "<parameter=path>\n123\n</parameter>\n",
    42: "<parameter=lines>\n7\n</parameter>\n",
    43: '<parameter=range>\n{"start": 1}\n</parameter>\n',
    44: "<parameter=flags>\n[true, null]\n</parameter>\n",
    45: "<parameter=exact>\nfalse\n</parameter>\n",
    46: "<parameter=extra>\n[1]\n</parameter>\n",
    47: "<parameter=note>\nnull\n</parameter>\n",
    48: '<parameter=path>\n"quoted"\n</parameter>\n',
    49: "<parameter=lines>\nNaN\n</parameter>\n",
    50: "<parameter=lines>\nseven\n</parameter>\n",
    51: "<parameter=note>\ntrue\n</parameter>\n",
}


def _arguments(*parameter_ids: int, tools=(READ_FILE_TOOL,)) -> str:
    """The arguments string of one read_file call assembled from ``parameter_ids`` under ``tools``."""

    assembler = Qwen38ReplyAssembler(_decoder({**TOOL_BLOCK, **ARGUMENT_PIECES}), thinking_open=False, tools=tools)
    _run(assembler, [TOOL_CALL_START_ID, 40, *parameter_ids, 13, TOOL_CALL_END_ID])
    assert assembler.parse_errors == [] and len(assembler.calls) == 1
    return assembler.calls[0]["function"]["arguments"]


def test_assembler_types_tool_arguments_by_the_schema() -> None:
    # A string parameter is verbatim whatever it looks like; the JSON-typed ones are parsed; an undeclared one is
    # parsed when it is JSON.
    assert json.loads(_arguments(41, 42, 43, 44, 45, 46)) == {
        "path": "123",
        "lines": 7,
        "range": {"start": 1},
        "flags": [True, None],
        "exact": False,
        "extra": [1],
    }
    assert json.loads(_arguments(48)) == {"path": '"quoted"'}
    # ["string", "null"]: null is null, anything else is the text.
    assert json.loads(_arguments(47)) == {"note": None} and json.loads(_arguments(51)) == {"note": "true"}
    # NaN is not JSON: it stays text (the wire stays valid JSON); a non-JSON body for a typed parameter stays text.
    assert _arguments(49) == '{"lines": "NaN"}' and json.loads(_arguments(50)) == {"lines": "seven"}
    # Without the schema (a tool the request did not offer) every JSON-looking value is parsed, as before.
    assert json.loads(_arguments(41, tools=())) == {"path": 123}
    assert json.loads(_arguments(41, tools=(ADD_TOOL,))) == {"path": 123}


def test_assembler_stop_strings_hold_back_the_tail_and_exclude_the_match() -> None:
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False, stop_strings=("STOP",))
    first = assembler.push(21)
    assert first == [] and assembler.held_text == "Hel"  # shorter than the hold-back
    second = assembler.push(22)
    assert second == [{"content": "Hello"}] and assembler.held_text == " ST"
    third = assembler.push(23)
    assert third == [] and assembler.stop_hit is True
    assert assembler.push(24) == [] and assembler.finish() == []
    assert "".join(assembler.content) == "Hello"
    # Without a hit the held tail is flushed at the end, trailing whitespace dropped.
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=False, stop_strings=("STOP",))
    deltas = _run(assembler, [21, 24, 15])
    assert "".join(delta["content"] for delta in deltas) == "Hel world" and deltas[0] == {"content": "Hel wo"}
    # Stop strings do not apply inside reasoning.
    assembler = Qwen38ReplyAssembler(_decoder(TOOL_BLOCK), thinking_open=True, stop_strings=("STOP",))
    deltas = _run(assembler, [22, 23, THINK_END_ID, 21])
    assert (
        "".join(assembler.reasoning) == "lo STOP" and deltas[-1] == {"content": "Hel"} and assembler.stop_hit is False
    )


def test_assembler_holds_a_split_utf8_sequence() -> None:
    pieces = {(1,): "�", (1, 2): "é", (3,): "x"}
    assembler = Qwen38ReplyAssembler(lambda ids: pieces[tuple(ids)], thinking_open=False)
    assert (
        assembler.push(1) == [] and assembler.push(2) == [{"content": "é"}] and assembler.push(3) == [{"content": "x"}]
    )


def test_assembler_flush_scans_the_held_and_pending_text_for_stop_strings() -> None:
    # The budget ends the reply while a split UTF-8 token is pending: its text completes the stop string.
    pieces = {(1,): "The E", (2,): "ND�"}
    assembler = Qwen38ReplyAssembler(lambda ids: pieces[tuple(ids)], thinking_open=False, stop_strings=("END",))
    assert assembler.push(1) == [{"content": "The"}] and assembler.held_text == " E" and assembler.push(2) == []
    assert assembler.finish() == [] and assembler.stop_hit is True and "".join(assembler.content) == "The"
    # A phase change flushes the same way.
    assembler = Qwen38ReplyAssembler(lambda ids: pieces[tuple(ids)], thinking_open=False, stop_strings=("END",))
    assembler.push(1)
    assembler.push(2)
    assert assembler.push(TOOL_CALL_START_ID) == [] and assembler.stop_hit is True
    assert "".join(assembler.content) == "The"


# -- checkpoint: template equality against transformers (4x p150) --------------------------------------

MATRIX_TOOLS = {
    "none": [],
    "one": [ADD_TOOL],
    "three_nested": [
        ADD_TOOL,
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "absolute path"},
                        "range": {
                            "type": "object",
                            "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                        },
                        "flags": {"type": "array", "items": {"enum": ["z", "a"], "type": "string"}},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
    ],
}
MATRIX_THINKING = {"off_low": (False, "low"), "on_xhigh": (True, "xhigh"), "on_medium": (True, "medium")}
SYSTEM = {"role": "system", "content": "You are Hermes, a careful agent."}
USER = {"role": "user", "content": "Add 2 and 3, then tell me the result."}
CALL = {
    "role": "assistant",
    "content": "I'll use the tool.",
    "tool_calls": [
        {"id": "call_a1", "type": "function", "function": {"name": "add_integers", "arguments": '{"a": 2, "b": 3}'}}
    ],
}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_a1", "name": "add_integers", "content": '{"a":2,"b":3,"sum":5}'}
MATRIX_MESSAGES = {
    "system_user": [SYSTEM, USER],
    "user_only": [USER],
    "with_assistant": [
        SYSTEM,
        USER,
        {"role": "assistant", "content": "Sure: 5."},
        {"role": "user", "content": "And 4 + 4?"},
    ],
    "tool_round": [SYSTEM, USER, CALL, TOOL_RESULT],
    "two_tool_results": [
        SYSTEM,
        USER,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                CALL["tool_calls"][0],
                {
                    "id": "call_b2",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/x", "range": {"start": 1, "end": 2}, "flags": ["a"]}',
                    },
                },
            ],
        },
        TOOL_RESULT,
        {"role": "tool", "tool_call_id": "call_b2", "content": "line 1\nline 2"},
    ],
    "null_content_call": [SYSTEM, USER, {**CALL, "content": None}, TOOL_RESULT, {"role": "user", "content": "thanks"}],
    "with_reasoning": [
        SYSTEM,
        USER,
        {"role": "assistant", "content": "5", "reasoning_content": "2 + 3 = 5.\n"},
        {"role": "user", "content": "ok"},
    ],
}


def _reference_messages(messages: list[dict]) -> list[dict]:
    """The raw request with only the string -> dict ``arguments`` conversion the reference template needs."""

    result = []
    for message in messages:
        copied = json.loads(json.dumps(message))
        for call in copied.get("tool_calls") or ():
            if isinstance(call["function"].get("arguments"), str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        result.append(copied)
    return result


def _reference_ids(template, messages, tools, enable_thinking, reasoning_effort) -> list[int]:
    encoded = template.tokenizer.apply_chat_template(
        _reference_messages(messages),
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        preserve_thinking=True,
        reasoning_effort=reasoning_effort,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    while ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


@pytest.fixture(scope="module")
def template() -> Qwen38OfficialChatTemplate:
    if not CHECKPOINT.is_dir():
        pytest.skip(f"pinned checkpoint is not present: {CHECKPOINT}")
    return Qwen38OfficialChatTemplate(CHECKPOINT)


def test_special_token_ids_match_the_tokenizer(template: Qwen38OfficialChatTemplate) -> None:
    expected = {
        "<think>": THINK_START_ID,
        "</think>": THINK_END_ID,
        "<tool_call>": TOOL_CALL_START_ID,
        "</tool_call>": TOOL_CALL_END_ID,
        "<|im_start|>": IM_START_ID,
        "<|im_end|>": IM_END_ID,
    }
    actual = {tag: template.tokenizer.convert_tokens_to_ids(tag) for tag in expected}
    assert actual == expected, f"tokenizer ids {actual} vs pinned {expected}"
    assert template.tokenizer("</think>", add_special_tokens=False).input_ids == [THINK_END_ID]
    assert template.tokenizer("<tool_call>\n", add_special_tokens=False).input_ids[0] == TOOL_CALL_START_ID
    assert template.tokenizer.decode([THINK_END_ID], skip_special_tokens=False) == "</think>"


@pytest.mark.parametrize("thinking", sorted(MATRIX_THINKING))
@pytest.mark.parametrize("tools", sorted(MATRIX_TOOLS))
@pytest.mark.parametrize("shape", sorted(MATRIX_MESSAGES))
def test_server_render_equals_transformers_reference_bitwise(template, shape, tools, thinking) -> None:
    # The server renders exactly the client's messages (no system prompt is added: user_only is the case without one).
    enable_thinking, effort = MATRIX_THINKING[thinking]
    messages = MATRIX_MESSAGES[shape]
    server_ids = protocol.render_prompt(
        template.tokenizer, messages, MATRIX_TOOLS[tools], enable_thinking=enable_thinking, reasoning_effort=effort
    )
    reference = _reference_ids(template, messages, MATRIX_TOOLS[tools], enable_thinking, effort)
    first = next((index for index, (a, b) in enumerate(zip(server_ids, reference)) if a != b), None)
    assert server_ids == reference, (
        f"{shape}/{tools}/{thinking}: server {len(server_ids)} ids vs reference {len(reference)}, first difference at "
        f"{first}: {server_ids[first - 3 if first else 0 : (first or 0) + 3]} vs {reference[first - 3 if first else 0 : (first or 0) + 3]}"
    )
    suffix = template.tokenizer(
        "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n", add_special_tokens=False
    ).input_ids
    assert server_ids[0] == IM_START_ID and server_ids[-len(suffix) :] == suffix


def test_sorted_tool_keys_would_break_equality(template: Qwen38OfficialChatTemplate) -> None:
    """The defect the key-order rule guards: re-serialising the tools with sort_keys changes the rendered text."""

    sorted_tools = json.loads(json.dumps(MATRIX_TOOLS["three_nested"], sort_keys=True))
    messages = [SYSTEM, USER]
    kept = protocol.render_prompt(
        template.tokenizer, messages, MATRIX_TOOLS["three_nested"], enable_thinking=False, reasoning_effort="low"
    )
    resorted = protocol.render_prompt(
        template.tokenizer, messages, sorted_tools, enable_thinking=False, reasoning_effort="low"
    )
    assert kept != resorted
    assert kept == _reference_ids(template, messages, MATRIX_TOOLS["three_nested"], False, "low")


@pytest.mark.parametrize(
    "messages, fragment",
    [
        ([SYSTEM, {"role": "robot", "content": "x"}], "role must be one of"),
        ([USER, SYSTEM], "must be first"),
        ([SYSTEM, TOOL_RESULT], "No user query"),
        (
            [SYSTEM, USER, {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{"}}]}],
            "not valid JSON",
        ),
    ],
)
def test_server_rejects_what_the_reference_raises(expect_error, template, messages, fragment) -> None:
    with expect_error(Qwen38ChatRequestRejected, match=fragment):
        protocol.render_prompt(template.tokenizer, messages, [], enable_thinking=True, reasoning_effort="medium")
    if fragment != "not valid JSON":  # the reference itself cannot render string arguments at all
        with expect_error(Exception):
            _reference_ids(template, messages, [], True, "medium")


def test_a_thinking_turns_reference_render_never_extends_the_committed_ids_on_the_real_tokenizer(
    template: Qwen38OfficialChatTemplate,
) -> None:
    """Turn N with tools and thinking on, the model's reasoning and tool-call text as generated, the tool result
    appended for turn N+1.  The device prompt is the reference render of that history (decision A: the reasoning is
    not carried forward), and on the real tokenizer it never extends the committed ids: the template renders the
    past turn's think block empty and the "\\n" after <think> merges into "\\n\\n", so the common prefix stops one
    token before the served prompt's end and the next turn is a full prefill.  With the reasoning echoed the render
    agrees for string and integer arguments and misses for nested ones (a tojson re-serialisation)."""

    tokenizer = template.tokenizer
    flags = {"enable_thinking": True, "reasoning_effort": "medium"}
    tools = MATRIX_TOOLS["three_nested"]
    messages = protocol.normalize_messages([SYSTEM, USER])
    prompt = protocol.render_prompt(tokenizer, messages, tools, **flags)
    cases = {
        "string_and_int": (
            "<function=add_integers>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>",
            True,
        ),
        "nested": (
            '<function=read_file>\n<parameter=path>\n/tmp/x\n</parameter>\n<parameter=range>\n{"start":1,"end":2}\n</parameter>\n</function>',
            False,
        ),
    }
    for name, (call_text, reference_agrees) in cases.items():
        generated_text = "Thinking briefly.\n</think>\n\nI'll check.\n\n<tool_call>\n" + call_text + "\n</tool_call>"
        generated = tokenizer(generated_text, add_special_tokens=False).input_ids + [IM_END_ID]
        assembler = Qwen38ReplyAssembler(
            lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            thinking_open=True,
        )
        for token in generated[:-1]:
            assembler.push(token)
        assembler.finish()
        assert len(assembler.calls) == 1 and assembler.parse_errors == [], name
        committed = prompt + generated
        echo = {k: v for k, v in assembler.message().items() if k != "reasoning_content"}
        history = protocol.normalize_messages([*messages, echo, {"role": "tool", "content": "result"}])
        full = protocol.render_prompt(tokenizer, history, tools, **flags)
        common = 0
        while common < len(full) and common < len(committed) and full[common] == committed[common]:
            common += 1
        assert len(prompt) - 1 <= common < len(prompt), (name, common, len(prompt))
        assert full[common - 1 : common + 1] == [
            THINK_START_ID,
            tokenizer("\n\n", add_special_tokens=False).input_ids[0],
        ]
        assert committed[common - 1 : common + 1] == [
            THINK_START_ID,
            tokenizer("\n", add_special_tokens=False).input_ids[0],
        ]
        echoed = protocol.normalize_messages([*messages, assembler.message(), {"role": "tool", "content": "result"}])
        full_with_reasoning = protocol.render_prompt(tokenizer, echoed, tools, **flags)
        assert (full_with_reasoning[: len(committed)] == committed) is reference_agrees, name
