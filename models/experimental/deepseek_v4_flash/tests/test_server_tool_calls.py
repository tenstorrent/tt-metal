# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the server's DSML tool-call handling.

DeepSeek-V4 writes tool calls as DSML (DeepSeek Markup Language), the XML-like block
defined by the checkpoint's own ``encoding/encoding_dsv4.py``. The server has to keep
that markup out of the text a reader sees and turn it into OpenAI ``tool_calls``.

The streamer is driven one character at a time here, which is the worst case: every tag
is split across chunks, so anything that leaks markup or emits a delta mid-tag fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.experimental.deepseek_v4_flash.demo import server as S

BLOCK = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
<｜DSML｜parameter name="days" string="false">5</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""


class _Tokenizer:
    """Stands in for the real tokenizer: ``push(n)`` reveals the first n characters."""

    def __init__(self, reply: str):
        self.reply = reply

    def decode(self, count, skip_special_tokens=False) -> str:
        return self.reply[:count]


def run(reply: str) -> tuple[S._Streamer, str, str]:
    """Stream ``reply`` a character at a time; return the streamer and what was sent."""
    sent: list[tuple[str, str]] = []
    streamer = S._Streamer(_Tokenizer(reply), lambda reasoning, content: sent.append((reasoning, content)))
    for count in range(1, len(reply) + 1):
        streamer.push(count)
    streamer.close()
    return streamer, "".join(r for r, _ in sent), "".join(c for _, c in sent)


def test_the_block_never_reaches_the_reader() -> None:
    streamer, _, streamed = run(f"Let me check that.\n\n{BLOCK}")

    assert streamer.content == "Let me check that."
    assert streamed == "Let me check that.", "markup or the block's blank-line separator was streamed"
    assert "DSML" not in streamed and "DSML" not in streamer.content


def test_parameters_keep_the_types_their_string_flag_declares() -> None:
    streamer, _, _ = run(BLOCK)
    (call,) = streamer.tool_calls

    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    # string="true" is a raw string, string="false" is JSON: 5 must not arrive as "5".
    assert json.loads(call["function"]["arguments"]) == {"city": "San Francisco", "days": 5}


def test_a_reasoning_block_and_a_tool_call_coexist() -> None:
    streamer, reasoning, streamed = run(f"<think>The user wants weather.</think>On it.\n\n{BLOCK}")

    assert reasoning == "The user wants weather."
    assert streamer.content == "On it." and streamed == "On it."
    assert len(streamer.tool_calls) == 1


def test_several_invocations_are_all_reported() -> None:
    block = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="first">\n'
        '<｜DSML｜parameter name="a" string="true">x</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="second">\n'
        '<｜DSML｜parameter name="b" string="false">[1, 2]</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )
    streamer, _, _ = run(block)

    assert [c["function"]["name"] for c in streamer.tool_calls] == ["first", "second"]
    assert json.loads(streamer.tool_calls[1]["function"]["arguments"]) == {"b": [1, 2]}


def test_arguments_written_as_bare_json_are_read_too() -> None:
    """The model sometimes writes a JSON object inside invoke instead of parameter tags."""
    block = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="fn">\n{"city": "Paris"}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    streamer, _, _ = run(block)

    assert json.loads(streamer.tool_calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_a_value_flagged_json_but_written_as_text_is_kept() -> None:
    """Better to forward the call and let the tool reject it than to fail the request."""
    block = (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="fn">\n'
        '<｜DSML｜parameter name="n" string="false">not json</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
    streamer, _, _ = run(block)

    assert json.loads(streamer.tool_calls[0]["function"]["arguments"]) == {"n": "not json"}


def test_an_unclosed_block_still_hides_its_markup() -> None:
    """Hitting the token cap mid-block must not spill half a tag into the reply."""
    truncated = f"Checking.\n\n{BLOCK[: len(BLOCK) // 2]}"
    streamer, _, streamed = run(truncated)

    assert streamer.content == "Checking." and "DSML" not in streamed


def test_an_ordinary_reply_is_untouched() -> None:
    streamer, _, streamed = run("Two paragraphs.\n\nSecond one.")

    assert streamer.content == "Two paragraphs.\n\nSecond one."
    assert streamed == "Two paragraphs.\n\nSecond one.", "held-back newlines were never released"
    assert streamer.tool_calls == []


def test_no_tool_call_means_no_calls_parsed() -> None:
    streamer, _, _ = run("Just an answer.")
    assert streamer.tool_calls == []


TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object"}},
    }
]


def test_a_tool_result_is_accepted_and_folded_into_a_user_turn() -> None:
    """The error this fixes: agent clients send results back as ``role: "tool"``."""
    messages = S._normalized_messages(
        [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
        ]
    )

    # The model has no tool role: the result becomes the next user turn.
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content_blocks"] == [{"type": "tool_result", "tool_use_id": "call_1", "content": "18C"}]


def test_the_result_renders_as_a_tool_result_block() -> None:
    """What the model actually reads has to be the checkpoint's own wrapper."""
    from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message

    messages = S._normalized_messages(
        [{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "call_1", "content": "18C"}]
    )
    rendered = "".join(render_message(i, messages, "chat") for i in range(len(messages)))

    assert "<tool_result>18C</tool_result>" in rendered


def test_the_assistants_call_renders_back_as_dsml() -> None:
    """A follow-up re-renders the history, so the call must survive the round trip."""
    from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message

    messages = S._normalized_messages(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "18C"},
        ]
    )
    rendered = "".join(render_message(i, messages, "chat") for i in range(len(messages)))

    assert '<｜DSML｜invoke name="get_weather">' in rendered
    assert '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>' in rendered


def test_tools_are_rendered_into_the_system_turn() -> None:
    """Without the schemas in the prompt the model has nothing to call."""
    from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message

    messages = S._normalized_messages([{"role": "user", "content": "weather?"}], tools=TOOLS)
    assert messages[0]["role"] == "system", "a system turn is created to carry the schemas"

    rendered = "".join(render_message(i, messages, "chat") for i in range(len(messages)))
    assert "## Tools" in rendered and "get_weather" in rendered


def test_tools_join_an_existing_system_message() -> None:
    messages = S._normalized_messages(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"}], tools=TOOLS
    )

    assert [m["role"] for m in messages] == ["system", "user"], "no second system turn was invented"
    assert messages[0]["tools"] == TOOLS


def test_a_tool_loop_continues_on_the_warm_cache() -> None:
    """The whole point of storing the call: the follow-up must not re-prefill.

    The server stores its own reply, the client echoes it back with the result, and the
    stored conversation has to still read as a prefix of what arrived.
    """
    first = S._normalized_messages([{"role": "user", "content": "weather?"}], tools=TOOLS)
    stored = list(first)  # what the session holds after admitting the first turn
    stored.append(
        {
            "role": "assistant",
            "content": "checking",
            "reasoning_content": "the user wants weather",  # never echoed back by clients
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        }
    )

    follow_up = S._normalized_messages(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "18C"},
        ],
        tools=TOOLS,
    )

    assert S._continues(follow_up, stored), "the tool result forced a full re-prefill"


def test_an_unknown_role_is_refused_clearly(expect_error) -> None:
    with expect_error(S.RequestError, "no place for"):
        S._normalized_messages([{"role": "user", "content": "hi"}, {"role": "function", "content": "x"}])


def test_a_trailing_assistant_message_is_still_refused(expect_error) -> None:
    """Only the user or a tool can end a request; there would be nothing to answer."""
    with expect_error(S.RequestError, "user or a tool"):
        S._normalized_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}])


REFERENCE = next(
    (
        path
        for path in Path("/home/ttuser/.cache/huggingface/hub").glob(
            "models--deepseek-ai--DeepSeek-V4-Flash*/snapshots/*/encoding"
        )
    ),
    None,
)


@pytest.mark.skipif(REFERENCE is None, reason="the DeepSeek-V4-Flash checkpoint is not on this host")
def test_round_trips_the_checkpoints_own_encoder() -> None:
    """Guards against the format drifting: the block is built by the model's encoder.

    Hand-written fixtures only prove the parser matches what this file assumed, so the
    tags come from ``encoding_dsv4.py`` in the checkpoint itself.
    """
    import sys

    sys.path.insert(0, str(REFERENCE))
    import encoding_dsv4 as reference

    arguments = {"city": "San Francisco", "days": 5, "metric": True, "tags": ["a", "b"]}
    call = {"name": "get_weather", "arguments": json.dumps(arguments)}
    block = reference.tool_calls_template.format(
        dsml_token=reference.dsml_token,
        tc_block_name=reference.tool_calls_block_name,
        tool_calls=reference.tool_call_template.format(
            dsml_token=reference.dsml_token,
            name=call["name"],
            arguments=reference.encode_arguments_to_dsml(call),
        ),
    )

    streamer, _, streamed = run(f"Checking.\n\n{block}")

    assert streamed == "Checking."
    assert json.loads(streamer.tool_calls[0]["function"]["arguments"]) == arguments
