# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for Muse-Glimmer's ATEM tool parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from models.autoports.meta_models_muse_glimmer_30b.tt.muse_glimmer_tool_parser import (
    MuseGlimmerToolParser,
    ToolParserManager,
    _decode_value,
    _iter_messages,
)

WEIGHT_REVISION = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"


def request(*, tool_choice="auto", names=("get_weather",)) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="meta-models/Muse-Glimmer-30B",
        messages=[{"role": "user", "content": "use a tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Call {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in names
        ],
        tool_choice=tool_choice,
    )


def invoke(name: str, **params: str) -> str:
    body = "".join(f'<atem:parameter name="{key}">{value}</atem:parameter>\n' for key, value in params.items())
    return "<atem:function_calls>\n" f'<atem:invoke name="{name}">\n{body}</atem:invoke>\n' "</atem:function_calls>"


def message(recipient: str | None, body: str, end: str = "<|eom|>") -> str:
    target = f" to={recipient}" if recipient else ""
    return f"<|start|>assistant{target}<|message|>{body}{end}"


def tool_message(name: str, body: str, end: str = "<|eom|>") -> str:
    return message(name, body, end)


@pytest.fixture
def parser() -> MuseGlimmerToolParser:
    return MuseGlimmerToolParser(tokenizer=None)


def extract(parser: MuseGlimmerToolParser, text: str, req=None):
    return parser.extract_tool_calls(text, req or request())


def call_arguments(call) -> dict:
    return json.loads(call.function.arguments)


def feed(parser, chunks, req=None):
    """Drive vLLM's cumulative streaming parser API."""
    previous = ""
    deltas = []
    req = req or request()
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(previous, current, chunk, [], [], [], req)
        if delta is not None:
            deltas.append(delta)
        previous = current
    return deltas


# Message segmentation and channel selection.


def test_iter_messages_recognizes_bare_first_header_and_closed_state():
    text = " to=self<|message|>thinking<|eom|>" + tool_message("get_weather", invoke("get_weather", city="Paris"))
    messages = list(_iter_messages(text))
    assert messages[0] == ("self", "thinking", True)
    assert messages[1][0] == "get_weather"
    assert messages[1][2] is True


def test_next_header_terminates_a_damaged_reasoning_message(parser):
    text = message("self", "unfinished reasoning", end="") + tool_message(
        "get_weather", invoke("get_weather", city="Paris")
    )
    out = extract(parser, text)
    assert out.tools_called is True
    assert call_arguments(out.tool_calls[0]) == {"city": "Paris"}


@pytest.mark.parametrize("recipient", ["self", "user"])
def test_atem_markup_in_non_tool_channels_is_never_dispatched(parser, recipient):
    echoed = message(recipient, invoke("delete_file", path="important.py"))
    out = extract(parser, echoed, request(names=("delete_file",)))
    assert out.tools_called is False
    assert out.tool_calls == []


def test_unframed_atem_fallback_still_extracts(parser):
    out = extract(parser, invoke("get_weather", city="Paris"))
    assert out.tools_called is True
    assert call_arguments(out.tool_calls[0]) == {"city": "Paris"}


# Tool extraction, normalization, and malformed output.


def test_single_call_and_json_value_decoding(parser):
    raw = tool_message(
        "search",
        invoke(
            "search",
            query="two sorted lists",
            limit="10",
            metric="true",
            tags='["a", "b"]',
            opts='{"deep": true}',
        ),
    )
    out = extract(parser, raw, request(names=("search",)))
    assert out.tools_called is True
    assert out.tool_calls[0].function.name == "search"
    assert call_arguments(out.tool_calls[0]) == {
        "query": "two sorted lists",
        "limit": 10,
        "metric": True,
        "tags": ["a", "b"],
        "opts": {"deep": True},
    }


def test_non_json_and_whitespace_are_preserved():
    assert _decode_value("  padded value  ") == "  padded value  "
    assert _decode_value("line one\nline two\n") == "line one\nline two\n"


def test_parallel_calls_keep_emission_order(parser):
    body = invoke("read_file", path="a.py") + invoke("read_file", path="b.py")
    out = extract(parser, tool_message("read_file", body), request(names=("read_file",)))
    assert [call.function.name for call in out.tool_calls] == ["read_file", "read_file"]
    assert [call_arguments(call) for call in out.tool_calls] == [
        {"path": "a.py"},
        {"path": "b.py"},
    ]


def test_doubled_bare_name_is_normalized_to_registered_name(parser):
    out = extract(
        parser,
        tool_message("get_weather", invoke("get_weather.get_weather", city="Paris")),
    )
    assert out.tool_calls[0].function.name == "get_weather"


def test_namespaced_registered_name_is_preserved(parser):
    req = request(names=("filesystem.read_file",))
    out = extract(
        parser,
        tool_message("filesystem.read_file", invoke("filesystem.read_file", path="a.py")),
        req,
    )
    assert out.tool_calls[0].function.name == "filesystem.read_file"


def test_ambiguous_leaf_name_is_not_rebound(parser):
    req = request(names=("calendar.get",))
    out = extract(parser, tool_message("weather.get", invoke("weather.get", city="Paris")), req)
    assert out.tool_calls[0].function.name == "weather.get"


def test_truncated_invoke_is_not_fabricated(parser):
    truncated = tool_message(
        "get_weather",
        '<atem:function_calls><atem:invoke name="get_weather">',
    )
    out = extract(parser, truncated)
    assert out.tools_called is False
    assert out.tool_calls == []


def test_content_reasoning_and_tool_call_are_kept_separate(parser):
    text = (
        message("self", "I should inspect the weather.")
        + tool_message("get_weather", invoke("get_weather", city="Paris"))
        + message("user", "The result is ready.", end="<|eot|>")
    )
    out = extract(parser, text)
    assert out.tools_called is True
    assert out.content == "The result is ready."
    assert "inspect" not in out.content


# Request adjustment.


@pytest.mark.parametrize("choice", ["auto", "required"])
def test_adjust_request_keeps_special_tokens_for_all_tool_choices(parser, choice):
    req = request(tool_choice=choice)
    parser.adjust_request(req)
    assert req.skip_special_tokens is False


def test_required_choice_does_not_install_incompatible_json_guidance(parser):
    req = request(tool_choice="required")
    parser.adjust_request(req)
    assert req.structured_outputs is None


def test_named_choice_does_not_install_incompatible_json_guidance(parser):
    req = request(
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        }
    )
    parser.adjust_request(req)
    assert req.skip_special_tokens is False
    assert req.structured_outputs is None


# Streaming.


def test_streaming_separates_reasoning_calls_and_final_content(parser):
    text = (
        message("self", "Think first.")
        + tool_message("get_weather", invoke("get_weather", city="Paris"))
        + message("user", "Done.", end="<|eot|>")
    )
    deltas = feed(parser, [text[i : i + 7] for i in range(0, len(text), 7)])
    reasoning = "".join(delta.reasoning or "" for delta in deltas)
    content = "".join(delta.content or "" for delta in deltas)
    calls = [call for delta in deltas for call in (delta.tool_calls or [])]
    assert reasoning == "Think first."
    assert content == "Done."
    assert len(calls) == 1
    assert calls[0].id
    assert calls[0].function.name == "get_weather"
    assert json.loads(calls[0].function.arguments) == {"city": "Paris"}


def test_streaming_never_leaks_split_structural_markers(parser):
    text = message("user", "Visible answer.", end="<|eot|>")
    deltas = feed(parser, list(text))
    content = "".join(delta.content or "" for delta in deltas)
    assert content == "Visible answer."
    assert "<|" not in content and "to=user" not in content


def test_streaming_parallel_calls_are_emitted_once(parser):
    body = invoke("read_file", path="a.py") + invoke("read_file", path="b.py")
    text = tool_message("read_file", body)
    deltas = feed(
        parser,
        [text[: len(text) // 2], text[len(text) // 2 :], "ignored"],
        request(names=("read_file",)),
    )
    calls = [call for delta in deltas for call in (delta.tool_calls or [])]
    assert len(calls) == 2
    assert [call.index for call in calls] == [0, 1]


# Registration and tokenizer/chat-template contract.


def test_parser_is_registered_under_manifest_name():
    assert ToolParserManager.get_tool_parser("muse_glimmer") is MuseGlimmerToolParser


def test_openai_tool_result_round_trip_through_vllm_chat_path():
    from transformers import AutoTokenizer
    from vllm.entrypoints.chat_utils import _postprocess_messages

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-models/Muse-Glimmer-30B",
        revision=WEIGHT_REVISION,
        local_files_only=True,
    )
    tools = request().tools
    parsed = extract(
        MuseGlimmerToolParser(tokenizer=None),
        tool_message("get_weather", invoke("get_weather", city="Paris")),
    ).tool_calls[0]
    conversation = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": parsed.function.name,
                        "arguments": parsed.function.arguments,
                    },
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "18C"},
    ]
    _postprocess_messages(conversation)
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    rendered = tokenizer.apply_chat_template(
        conversation,
        tools=[tool.model_dump(exclude_none=True) for tool in tools],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "to=get_weather" in rendered
    assert "<tool_output" in rendered
    assert '"name": "get_weather"' in rendered


def test_parser_file_is_in_the_packaged_model_allowlist():
    parser_path = Path(__file__).parents[1] / "tt" / "muse_glimmer_tool_parser.py"
    assert parser_path.is_file()
