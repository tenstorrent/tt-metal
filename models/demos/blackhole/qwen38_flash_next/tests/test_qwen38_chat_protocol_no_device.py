# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chat protocol without a device: request normalisation, reply assembly and, with the checkpoint, the
template-equality matrix: the server's prompt ids must equal ``tokenizer.apply_chat_template`` bitwise for every
message shape x tools x thinking case."""

from __future__ import annotations

import json

import pytest

from models.demos.blackhole.qwen38_flash_next.chat import IM_END_ID, IM_START_ID, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.tests.helpers.chat import (
    CHAT_SPECIAL_TEXT,
    CHAT_SPECIAL_TOKENS,
    ChatTemplateFakeTokenizer,
    chat_request_rejection,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import (
    THINK_END_ID,
    THINK_START_ID,
    TOOL_CALL_END_ID,
    TOOL_CALL_START_ID,
    Qwen38ChatRequestRejected,
    Qwen38ReplyAssembler,
)

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
    call = {"id": "call_1", "type": "function", "function": {"name": "add_integers", "arguments": '{"b": 3, "a": 2}'}}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}], "name": "sam"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
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
    rejected = chat_request_rejection(protocol.normalize_messages, messages)
    assert rejected.param == param and rejected.code == "bad_request" and fragment in str(rejected)


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
        assert chat_request_rejection(protocol.validate_tools, bad).param == param


def test_stop_effort_and_thinking_budget_rules(expect_error) -> None:
    assert protocol.validate_stop(None) == () and protocol.validate_stop("END") == ("END",)
    assert protocol.validate_stop(["a", "b"]) == ("a", "b")
    for bad in ("", ["a", ""], ["1", "2", "3", "4", "5"], 3):
        with expect_error(Qwen38ChatRequestRejected):
            protocol.validate_stop(bad)
    efforts = [protocol.effort_level(value) for value in ("minimal", "low", "medium", "high", "xhigh")]
    assert efforts == ["low", "low", "medium", "xhigh", "xhigh"]
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
    tokenizer = ChatTemplateFakeTokenizer()
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

PIECES = {
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
    30: "The tags are ",
    31: " and ",
    32: "; the reasoning ends here.",
    33: " ok",
    # read_file parameters
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
ADD_CALL = [TOOL_CALL_START_ID, 10, 11, 12, 13, TOOL_CALL_END_ID]  # add_integers(a=2, b=3)
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


def _decoder(pieces: dict[int, str]):
    table = {**CHAT_SPECIAL_TEXT, **pieces}
    return lambda ids: "".join(table[token] for token in ids)


def _assembled(tokens, *, pieces=PIECES, thinking_open=False, **options) -> tuple[Qwen38ReplyAssembler, list[dict]]:
    """An assembler over ``pieces`` fed ``tokens`` then finished, and every delta it emitted."""

    assembler = Qwen38ReplyAssembler(_decoder(pieces), thinking_open=thinking_open, **options)
    deltas = [delta for token in tokens for delta in assembler.push(token)]
    return assembler, deltas + assembler.finish()


def test_assembler_streams_content_and_trims_the_edges_the_template_trims() -> None:
    assembler, deltas = _assembled([1, 2, 3, 4, 5, 4], pieces={1: "\n\n", 2: "Hello", 3: " world", 4: "\n", 5: "!"})
    assert deltas == [{"content": "Hello"}, {"content": " world"}, {"content": "\n!"}]
    assert assembler.message() == {"role": "assistant", "content": "Hello world\n!"}
    assert assembler.reasoning_tokens == 0 and assembler.tokens == 6 and assembler.calls == []


def test_assembler_splits_reasoning_from_content_at_the_think_end_token() -> None:
    pieces = {1: "Let me", 2: " think.", 3: "\n\n", 4: "Answer", 5: "\n"}
    assembler, deltas = _assembled([1, 2, 5, THINK_END_ID, 3, 4, 5], pieces=pieces, thinking_open=True)
    assert deltas == [{"reasoning_content": "Let me"}, {"reasoning_content": " think."}, {"content": "Answer"}]
    assert assembler.reasoning_tokens == 3  # the reasoning tokens before </think>, whitespace included
    assert assembler.message() == {"role": "assistant", "content": "Answer", "reasoning_content": "Let me think."}
    # Non-thinking mode: the think tags are the answer's text (a quoted tag never opens a reasoning phase).
    assembler, deltas = _assembled([THINK_END_ID, 4, THINK_START_ID, 1, THINK_END_ID, 4], pieces=pieces)
    assert "".join(delta["content"] for delta in deltas) == "</think>Answer<think>Let me</think>Answer"
    assert assembler.reasoning == [] and assembler.reasoning_tokens == 0


def test_assembler_tag_ids_act_only_in_the_phase_that_expects_them() -> None:
    # (a) An answer that quotes the tags: no phase change, the literals stay in the content.
    assembler, _ = _assembled([30, THINK_END_ID, 30, THINK_START_ID, 31, THINK_END_ID, 33], thinking_open=True)
    assert assembler.message() == {
        "role": "assistant",
        "content": "The tags are <think> and </think> ok",
        "reasoning_content": "The tags are",
    }
    # (b) Think tags inside a tool block are the argument's text; the call is parsed whole.
    assembler, _ = _assembled(
        [TOOL_CALL_START_ID, 16, THINK_START_ID, THINK_END_ID, 19, TOOL_CALL_END_ID], tools=[ADD_TOOL]
    )
    assert assembler.parse_errors == [] and len(assembler.calls) == 1
    assert json.loads(assembler.calls[0]["function"]["arguments"]) == {"query": "<think></think>"}
    # (c) A tool block drafted inside the reasoning is reasoning: no call, nothing to execute, the answer follows.
    assembler, deltas = _assembled([14, *ADD_CALL, 32, THINK_END_ID, 14], thinking_open=True)
    assert assembler.calls == [] and all("tool_calls" not in delta for delta in deltas)
    reasoning = "".join(assembler.reasoning)
    assert reasoning.startswith("I'll add them.<tool_call>\n<function=add_integers>")
    assert reasoning.endswith("</function>\n</tool_call>; the reasoning ends here.")
    assert assembler.message()["content"] == "I'll add them."
    # A <tool_call> opened inside a tool block is that block's text: the block fails to parse as one call and is
    # returned raw rather than restarted.
    assembler, deltas = _assembled([TOOL_CALL_START_ID, 10, *ADD_CALL])
    assert assembler.calls == [] and len(assembler.parse_errors) == 1
    assert deltas[0]["content"].count("<tool_call>") == 2


def test_assembler_emits_a_completed_tool_call_as_one_delta_with_openai_fields() -> None:
    assembler, deltas = _assembled([14, 15, *ADD_CALL, 15])
    assert len(deltas) == 2 and deltas[0] == {"content": "I'll add them."}
    [call] = deltas[1]["tool_calls"]
    assert call["index"] == 0 and call["type"] == "function"
    assert call["id"].startswith("call_") and len(call["id"]) == 17
    assert call["function"] == {"name": "add_integers", "arguments": '{"a": 2, "b": 3}'}
    message = assembler.message()
    assert message["content"] == "I'll add them."
    assert message["tool_calls"] == [{k: v for k, v in call.items() if k != "index"}]


def test_assembler_numbers_parallel_calls_keeps_multiline_and_json_parameters_and_nulls_empty_content() -> None:
    assembler, deltas = _assembled([*ADD_CALL, 15, TOOL_CALL_START_ID, 16, 17, 18, 19, TOOL_CALL_END_ID])
    assert [delta["tool_calls"][0]["index"] for delta in deltas] == [0, 1]
    second = json.loads(deltas[1]["tool_calls"][0]["function"]["arguments"])
    assert second == {"query": '{"x": [1, 2]}\nline two'}  # not JSON as a whole: kept verbatim, multiline
    assert assembler.message()["content"] is None and len(assembler.message()["tool_calls"]) == 2
    # A JSON value alone in a parameter is parsed (nested object).
    assembler, _ = _assembled([TOOL_CALL_START_ID, 16, 17, 19, TOOL_CALL_END_ID])
    assert json.loads(assembler.calls[0]["function"]["arguments"]) == {"query": {"x": [1, 2]}}


def test_assembler_falls_back_to_raw_content_for_malformed_and_truncated_tool_calls() -> None:
    assembler, deltas = _assembled([TOOL_CALL_START_ID, 10, 20, 13, TOOL_CALL_END_ID])
    assert deltas == [{"content": "<tool_call>\n<function=add_integers>\n<parameter=a>\n</function>\n</tool_call>"}]
    assert assembler.calls == [] and len(assembler.parse_errors) == 1 and "malformed" in assembler.parse_errors[0]
    assembler, deltas = _assembled([14, TOOL_CALL_START_ID, 10, 11])
    assert deltas == [
        {"content": "I'll add them."},
        {"content": "<tool_call>\n<function=add_integers>\n<parameter=a>\n2\n</parameter>"},
    ]
    assert assembler.truncated_tool_call is True and assembler.calls == []


def test_assembler_types_tool_arguments_by_the_schema() -> None:
    def arguments(*parameter_ids: int, tools=(READ_FILE_TOOL,)) -> str:
        assembler, _ = _assembled([TOOL_CALL_START_ID, 40, *parameter_ids, 13, TOOL_CALL_END_ID], tools=tools)
        assert assembler.parse_errors == [] and len(assembler.calls) == 1
        return assembler.calls[0]["function"]["arguments"]

    # A string parameter is verbatim whatever it looks like; the JSON-typed ones are parsed; an undeclared one is
    # parsed when it is JSON.
    assert json.loads(arguments(41, 42, 43, 44, 45, 46)) == {
        "path": "123",
        "lines": 7,
        "range": {"start": 1},
        "flags": [True, None],
        "exact": False,
        "extra": [1],
    }
    assert json.loads(arguments(48)) == {"path": '"quoted"'}
    # ["string", "null"]: null is null, anything else is the text.
    assert json.loads(arguments(47)) == {"note": None} and json.loads(arguments(51)) == {"note": "true"}
    # NaN is not JSON: it stays text (the wire stays valid JSON); a non-JSON body for a typed parameter stays text.
    assert arguments(49) == '{"lines": "NaN"}' and json.loads(arguments(50)) == {"lines": "seven"}
    # Without the schema (a tool the request did not offer) every JSON-looking value is parsed.
    assert json.loads(arguments(41, tools=())) == {"path": 123}
    assert json.loads(arguments(41, tools=(ADD_TOOL,))) == {"path": 123}


def test_assembler_stop_strings_hold_back_the_tail_and_exclude_the_match() -> None:
    assembler = Qwen38ReplyAssembler(_decoder(PIECES), thinking_open=False, stop_strings=("STOP",))
    assert assembler.push(21) == [] and assembler.held_text == "Hel"  # shorter than the hold-back
    assert assembler.push(22) == [{"content": "Hello"}] and assembler.held_text == " ST"
    assert assembler.push(23) == [] and assembler.stop_hit is True
    assert assembler.push(24) == [] and assembler.finish() == []
    assert "".join(assembler.content) == "Hello"
    # Without a hit the held tail is flushed at the end, trailing whitespace dropped.
    _, deltas = _assembled([21, 24, 15], stop_strings=("STOP",))
    assert "".join(delta["content"] for delta in deltas) == "Hel world" and deltas[0] == {"content": "Hel wo"}
    # Stop strings do not apply inside reasoning.
    assembler, deltas = _assembled([22, 23, THINK_END_ID, 21], thinking_open=True, stop_strings=("STOP",))
    assert "".join(assembler.reasoning) == "lo STOP" and deltas[-1] == {"content": "Hel"}
    assert assembler.stop_hit is False


def test_assembler_holds_a_split_utf8_sequence() -> None:
    pieces = {(1,): "�", (1, 2): "é", (3,): "x"}
    assembler = Qwen38ReplyAssembler(lambda ids: pieces[tuple(ids)], thinking_open=False)
    assert assembler.push(1) == [] and assembler.push(2) == [{"content": "é"}]
    assert assembler.push(3) == [{"content": "x"}]


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


# -- checkpoint: template equality against transformers ---------------------------------------------------

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
ADD_TOOL_CALL = {
    "id": "call_a1",
    "type": "function",
    "function": {"name": "add_integers", "arguments": '{"a": 2, "b": 3}'},
}
CALL = {"role": "assistant", "content": "I'll use the tool.", "tool_calls": [ADD_TOOL_CALL]}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_a1", "name": "add_integers", "content": '{"a":2,"b":3,"sum":5}'}
READ_FILE_ARGUMENTS = '{"path": "/tmp/x", "range": {"start": 1, "end": 2}, "flags": ["a"]}'
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
                ADD_TOOL_CALL,
                {
                    "id": "call_b2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": READ_FILE_ARGUMENTS},
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


def _reference_ids(template, messages, tools, enable_thinking, reasoning_effort) -> list[int]:
    """``apply_chat_template`` on the raw request with only the string -> dict ``arguments`` conversion the reference
    template needs."""

    reference_messages = json.loads(json.dumps(messages))
    for message in reference_messages:
        for call in message.get("tool_calls") or ():
            if isinstance(call["function"].get("arguments"), str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    encoded = template.tokenizer.apply_chat_template(
        reference_messages,
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


def test_special_token_ids_match_the_tokenizer(chat_template: Qwen38OfficialChatTemplate) -> None:
    tokenizer = chat_template.tokenizer
    actual = {tag: tokenizer.convert_tokens_to_ids(tag) for tag in CHAT_SPECIAL_TOKENS}
    assert actual == CHAT_SPECIAL_TOKENS, f"tokenizer ids {actual} vs pinned {CHAT_SPECIAL_TOKENS}"
    assert tokenizer("</think>", add_special_tokens=False).input_ids == [THINK_END_ID]
    assert tokenizer("<tool_call>\n", add_special_tokens=False).input_ids[0] == TOOL_CALL_START_ID
    assert tokenizer.decode([THINK_END_ID], skip_special_tokens=False) == "</think>"


@pytest.mark.parametrize("thinking", sorted(MATRIX_THINKING))
@pytest.mark.parametrize("tools", sorted(MATRIX_TOOLS))
@pytest.mark.parametrize("shape", sorted(MATRIX_MESSAGES))
def test_server_render_equals_transformers_reference_bitwise(chat_template, shape, tools, thinking) -> None:
    # The server renders exactly the client's messages (no system prompt is added: user_only is the case without one).
    enable_thinking, effort = MATRIX_THINKING[thinking]
    messages = MATRIX_MESSAGES[shape]
    server_ids = protocol.render_prompt(
        chat_template.tokenizer, messages, MATRIX_TOOLS[tools], enable_thinking=enable_thinking, reasoning_effort=effort
    )
    reference = _reference_ids(chat_template, messages, MATRIX_TOOLS[tools], enable_thinking, effort)
    first = next((index for index, (a, b) in enumerate(zip(server_ids, reference)) if a != b), None)
    window = slice(first - 3 if first else 0, (first or 0) + 3)
    assert server_ids == reference, (
        f"{shape}/{tools}/{thinking}: server {len(server_ids)} ids vs reference {len(reference)}, first difference at "
        f"{first}: {server_ids[window]} vs {reference[window]}"
    )
    suffix = chat_template.tokenizer(
        "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n", add_special_tokens=False
    ).input_ids
    assert server_ids[0] == IM_START_ID and server_ids[-len(suffix) :] == suffix


def test_sorted_tool_keys_would_break_equality(chat_template: Qwen38OfficialChatTemplate) -> None:
    """The defect the key-order rule guards: re-serialising the tools with sort_keys changes the rendered text."""

    tools = MATRIX_TOOLS["three_nested"]
    flags = {"enable_thinking": False, "reasoning_effort": "low"}
    kept = protocol.render_prompt(chat_template.tokenizer, [SYSTEM, USER], tools, **flags)
    resorted = protocol.render_prompt(
        chat_template.tokenizer, [SYSTEM, USER], json.loads(json.dumps(tools, sort_keys=True)), **flags
    )
    assert kept != resorted
    assert kept == _reference_ids(chat_template, [SYSTEM, USER], tools, False, "low")


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
def test_server_rejects_what_the_reference_raises(expect_error, chat_template, messages, fragment) -> None:
    with expect_error(Qwen38ChatRequestRejected, match=fragment):
        protocol.render_prompt(chat_template.tokenizer, messages, [], **FLAGS)
    if fragment != "not valid JSON":  # the reference itself cannot render string arguments at all
        with expect_error(Exception):
            _reference_ids(chat_template, messages, [], True, "medium")


@pytest.mark.parametrize(
    "call_text, reference_agrees",
    [
        ("<function=add_integers>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>", True),
        (
            "<function=read_file>\n<parameter=path>\n/tmp/x\n</parameter>\n"
            '<parameter=range>\n{"start":1,"end":2}\n</parameter>\n</function>',
            False,
        ),
    ],
    ids=["string_and_int", "nested"],
)
def test_a_thinking_turns_reference_render_never_extends_the_committed_ids_on_the_real_tokenizer(
    chat_template: Qwen38OfficialChatTemplate, call_text, reference_agrees
) -> None:
    """Turn N with tools and thinking on, then the tool result for turn N+1: the reference render of that history
    (the reasoning not carried forward) stops one token short of the committed ids, since the past think block
    renders empty, so the next turn is a full prefill. With the reasoning echoed the render agrees for string and
    integer arguments and misses for nested ones (a tojson re-serialisation)."""

    tokenizer = chat_template.tokenizer
    tools = MATRIX_TOOLS["three_nested"]
    messages = protocol.normalize_messages([SYSTEM, USER])
    prompt = protocol.render_prompt(tokenizer, messages, tools, **FLAGS)
    generated_text = "Thinking briefly.\n</think>\n\nI'll check.\n\n<tool_call>\n" + call_text + "\n</tool_call>"
    generated = tokenizer(generated_text, add_special_tokens=False).input_ids + [IM_END_ID]
    assembler = Qwen38ReplyAssembler(
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        thinking_open=True,
    )
    for token in generated[:-1]:
        assembler.push(token)
    assembler.finish()
    assert len(assembler.calls) == 1 and assembler.parse_errors == []
    committed = prompt + generated
    echo = {k: v for k, v in assembler.message().items() if k != "reasoning_content"}
    full = protocol.render_prompt(
        tokenizer, protocol.normalize_messages([*messages, echo, {"role": "tool", "content": "result"}]), tools, **FLAGS
    )
    common = 0
    while common < len(full) and common < len(committed) and full[common] == committed[common]:
        common += 1
    assert len(prompt) - 1 <= common < len(prompt), (common, len(prompt))
    assert full[common - 1 : common + 1] == [THINK_START_ID, tokenizer("\n\n", add_special_tokens=False).input_ids[0]]
    assert committed[common - 1 : common + 1] == [
        THINK_START_ID,
        tokenizer("\n", add_special_tokens=False).input_ids[0],
    ]
    echoed = protocol.normalize_messages([*messages, assembler.message(), {"role": "tool", "content": "result"}])
    full_with_reasoning = protocol.render_prompt(tokenizer, echoed, tools, **FLAGS)
    assert (full_with_reasoning[: len(committed)] == committed) is reference_agrees
