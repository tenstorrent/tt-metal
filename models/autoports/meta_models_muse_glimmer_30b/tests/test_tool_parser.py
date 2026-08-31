# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parsing for Muse-Glimmer-30B. No device, no weights.

The grammar under test is the one the checkpoint's own chat template instructs
the model to emit; ``test_template_example_parses`` pins that by parsing the
worked example the template itself shows the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EXT = Path(__file__).resolve().parent.parent / "vllm_ext"
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))

from muse_glimmer_vllm_ext.tool_parser import (  # noqa: E402
    BLOCK_CLOSE,
    BLOCK_OPEN,
    MuseGlimmerToolParser,
    parse_function_calls,
)


def block(*invokes: str) -> str:
    return BLOCK_OPEN + "\n" + "\n".join(invokes) + "\n" + BLOCK_CLOSE


def invoke(name: str, **params: str) -> str:
    body = "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params.items())
    return f'<atem:invoke name="{name}">\n{body}</atem:invoke>'


@pytest.fixture
def parser():
    return MuseGlimmerToolParser(tokenizer=None)


# ------------------------------------------------------------------ extraction


def test_single_call():
    calls = parse_function_calls(block(invoke("get_weather", city="Paris")))
    assert calls == [("get_weather", {"city": "Paris"})]


def test_mixed_value_types():
    raw = block(
        invoke(
            "search",
            query="two sorted lists",
            limit="10",
            metric="true",
            tags='["a","b"]',
            opts='{"deep": true}',
        )
    )
    ((_, args),) = parse_function_calls(raw)
    assert args == {
        "query": "two sorted lists",
        "limit": 10,
        "metric": True,
        "tags": ["a", "b"],
        "opts": {"deep": True},
    }


def test_parallel_calls_in_one_block():
    raw = block(invoke("read_file", path="a.py"), invoke("read_file", path="b.py"))
    assert parse_function_calls(raw) == [
        ("read_file", {"path": "a.py"}),
        ("read_file", {"path": "b.py"}),
    ]


def test_namespaced_name_is_passed_through():
    ((name, _),) = parse_function_calls(block(invoke("fs.read_file", path="x")))
    assert name == "fs.read_file", "namespace resolution is the scaffold's job"


def test_spaces_are_not_stripped():
    """The template states values are not stripped, so they must survive verbatim."""
    raw = block(invoke("echo", text="  padded  "))
    ((_, args),) = parse_function_calls(raw)
    assert args["text"] == "  padded  "


def test_multiline_value_keeps_its_newline():
    value = "line one\nline two\n"
    raw = (
        BLOCK_OPEN
        + (
            '\n<atem:invoke name="write">\n'
            f'<atem:parameter name="body">{value}</atem:parameter>\n'
            "</atem:invoke>\n"
        )
        + BLOCK_CLOSE
    )
    ((_, args),) = parse_function_calls(raw)
    assert args["body"] == value


def test_nan_stays_a_string():
    """json.loads accepts NaN; a parameter meaning the text "NaN" must not become a float."""
    ((_, args),) = parse_function_calls(block(invoke("f", x="NaN")))
    assert args["x"] == "NaN" and isinstance(args["x"], str)


# ------------------------------------------------------- ExtractedToolCallInfo


def test_no_tool_call_returns_content(parser):
    out = parser.extract_tool_calls("Just prose.", request=None)
    assert out.tools_called is False and out.content == "Just prose."


def test_prose_before_block_becomes_content(parser):
    text = "I'll check that.\n" + block(invoke("get_weather", city="Paris"))
    out = parser.extract_tool_calls(text, request=None)
    assert out.tools_called is True
    assert out.content.strip() == "I'll check that."
    assert out.tool_calls[0].function.name == "get_weather"
    assert json.loads(out.tool_calls[0].function.arguments) == {"city": "Paris"}


def test_unterminated_block_is_not_a_tool_call(parser):
    """A half-emitted block must surface as content, not a fabricated call."""
    text = BLOCK_OPEN + '\n<atem:invoke name="get_weather">\n'
    out = parser.extract_tool_calls(text, request=None)
    assert out.tools_called is False
    assert out.content == text


def test_arguments_are_a_json_string(parser):
    out = parser.extract_tool_calls(block(invoke("f", a="1")), request=None)
    args = out.tool_calls[0].function.arguments
    assert isinstance(args, str) and json.loads(args) == {"a": 1}


# ----------------------------------------------------------------- streaming


def feed(parser, chunks):
    """Drive the streaming API chunk by chunk; collect content and tool calls."""
    content, tool_calls, prev = [], [], ""
    for chunk in chunks:
        cur = prev + chunk
        delta = parser.extract_tool_calls_streaming(prev, cur, chunk, [], [], [], None)
        if delta is not None:
            if delta.content:
                content.append(delta.content)
            if delta.tool_calls:
                tool_calls.extend(delta.tool_calls)
        prev = cur
    return "".join(content), tool_calls


def test_streaming_plain_content(parser):
    content, calls = feed(parser, ["Hello", " ", "world"])
    assert content == "Hello world" and calls == []


def test_streaming_emits_calls_once_complete(parser):
    raw = block(invoke("get_weather", city="Paris"))
    content, calls = feed(parser, ["Checking. "] + [raw[i : i + 7] for i in range(0, len(raw), 7)])
    assert content == "Checking. "
    assert len(calls) == 1
    assert calls[0].function.name == "get_weather"
    assert json.loads(calls[0].function.arguments) == {"city": "Paris"}


def test_streaming_never_leaks_a_split_open_tag(parser):
    """The open tag arriving in pieces must not be streamed as content."""
    raw = block(invoke("f", a="1"))
    content, calls = feed(parser, [raw[i : i + 3] for i in range(0, len(raw), 3)])
    assert BLOCK_OPEN[:5] not in content
    assert content == ""
    assert len(calls) == 1


def test_streaming_emits_calls_only_once(parser):
    raw = block(invoke("f", a="1"))
    _, calls = feed(parser, [raw, " trailing"])
    assert len(calls) == 1


# ------------------------------------------------ contract with the checkpoint


def test_parser_is_registered_under_its_name():
    import muse_glimmer_vllm_ext.tool_parser  # noqa: F401
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    assert ToolParserManager.get_tool_parser("muse_glimmer") is MuseGlimmerToolParser


def test_template_example_parses():
    """The worked example the template shows the model must parse to that example."""
    example = (
        "<atem:function_calls>\n"
        '<atem:invoke name="example_tool_name.example_function_name">\n'
        '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
        '<atem:parameter name="example_parameter_2">This is the value for the second parameter\n'
        'that can span\n"multiple" lines\n</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    ((name, args),) = parse_function_calls(example)
    assert name == "example_tool_name.example_function_name"
    assert args["example_parameter_1"] == "value_1"
    assert args["example_parameter_2"].startswith("This is the value for the second parameter")
    assert '"multiple" lines' in args["example_parameter_2"]


# ------------------------------------- multi-turn round trip, through vLLM's path


def test_openai_json_string_arguments_round_trip():
    """A client echoing our tool_calls back must render, not raise.

    The chat template *requires* ``function.arguments`` to be a dict and says so
    loudly: "a JSON string cannot be parsed in the HF jinja sandbox" (jinja has
    ``tojson`` but no inverse). OpenAI clients send a JSON string. vLLM bridges
    that in ``chat_utils._postprocess_messages``, so the round trip works through
    the server even though calling the tokenizer directly with a JSON string
    raises. This test goes through vLLM's own conversion so a regression there -
    or a template revision that stops accepting dicts - fails here rather than
    at the second turn of a live agent session.
    """
    from transformers import AutoTokenizer
    from vllm.entrypoints.chat_utils import _postprocess_messages

    tok = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B", local_files_only=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    # Exactly what an OpenAI client posts back after receiving our tool call.
    conversation = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "18C"},
    ]
    _postprocess_messages(conversation)
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Paris"
    }, "vLLM must hand the template a dict"

    rendered = tok.apply_chat_template(conversation, tools=tools, tokenize=False, add_generation_prompt=True)
    assert "to=get_weather" in rendered
    assert "<tool_output" in rendered
    assert '"name": "get_weather"' in rendered


def test_our_output_feeds_back_in():
    """Parse a model tool call, then round-trip it as an assistant turn."""
    from transformers import AutoTokenizer
    from vllm.entrypoints.chat_utils import _postprocess_messages

    parser = MuseGlimmerToolParser(tokenizer=None)
    extracted = parser.extract_tool_calls(block(invoke("get_weather", city="Paris")), request=None)
    call = extracted.tool_calls[0]

    conversation = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "18C"},
    ]
    _postprocess_messages(conversation)
    tok = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B", local_files_only=True)
    rendered = tok.apply_chat_template(conversation, tools=[], tokenize=False, add_generation_prompt=True)
    assert "to=get_weather" in rendered and "<tool_output" in rendered
