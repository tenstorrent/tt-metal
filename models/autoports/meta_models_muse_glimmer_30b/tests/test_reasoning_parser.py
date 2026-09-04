# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only acceptance tests for the Muse Glimmer vLLM reasoning parser.

No device, no server, no vLLM engine: the parser is pure text/token bookkeeping
over the model's channel format, and that is exactly what these pin.  The
strings below are verbatim shapes taken from the live autoport server
(``doc/tti_release/smoke/``), including the one that made this parser necessary
-- an instruction-following prompt whose analysis channel violates every
constraint the instruction sets before the ``user`` channel obeys them all.

Run::

    pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_reasoning_parser.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm.reasoning", reason="needs the serving vLLM install")

from models.autoports.meta_models_muse_glimmer_30b.tt.reasoning_parser import (  # noqa: E402
    MuseGlimmerReasoningParser,
    reached_visible_channel,
    split_channels,
)

# The photosynthesis smoke, abridged.  The analysis channel is full of capital
# letters and quoted fragments; the reply obeys "all lowercase".
REASONED = (
    " to=selfWrite a 4 sentence summary of photosynthesis. Entire response"
    " should be in English, and in all lowercase letters.\n\nWe need 4 sentences."
    " Must avoid capital.\n\nFinal answer.assistant to=userphotosynthesis is the"
    " process by which plants convert light energy into chemical energy."
)
REASONED_ANALYSIS = (
    "Write a 4 sentence summary of photosynthesis. Entire response should be in"
    " English, and in all lowercase letters.\n\nWe need 4 sentences. Must avoid"
    " capital.\n\nFinal answer."
)
REASONED_REPLY = "photosynthesis is the process by which plants convert light energy into" " chemical energy."

# A prompt the model answered directly, with no analysis channel at all.
DIRECT = " to=userSure! Here's a simple breakdown of supervised learning."
DIRECT_REPLY = "Sure! Here's a simple breakdown of supervised learning."

FULL_REASONED = " to=self<|message|>Think first.<|eom|>" "<|start|>assistant to=user<|message|>Done.<|eot|>"


class _FakeTokenizer:
    """Just enough tokenizer for the parser: a vocab and a decode()."""

    _SPECIALS = {"<|eom|>": 200007, "<|eot|>": 200008}

    def __init__(self, head_text: str = ""):
        self._head_text = head_text

    def get_vocab(self):
        return dict(self._SPECIALS)

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return self._head_text


class _LifecycleTokenizer:
    """Token-level fixture for vLLM's prompt-to-stream parser lifecycle."""

    _SPECIALS = {
        "<|eom|>": 200007,
        "<|eot|>": 200008,
        "<|start|>": 200022,
        "<|message|>": 200023,
    }
    _PIECES = {
        100: "A user prompt.",
        328: " to",
        19669: "=self",
        76221: "Think first.",
        140680: "assistant",
        76976: "=user",
        30550: "Done.",
    }

    def get_vocab(self):
        return dict(self._SPECIALS)

    def decode(self, ids, skip_special_tokens=True):
        specials_by_id = {token_id: token for token, token_id in self._SPECIALS.items()}
        text = []
        for token_id in ids:
            if token_id in specials_by_id:
                if not skip_special_tokens:
                    text.append(specials_by_id[token_id])
            else:
                text.append(self._PIECES[token_id])
        return "".join(text)


def _parser(head_text: str = "") -> MuseGlimmerReasoningParser:
    return MuseGlimmerReasoningParser(_FakeTokenizer(head_text))


def test_parser_is_eagerly_registered_for_file_path_plugins():
    from vllm.reasoning import ReasoningParserManager

    assert ReasoningParserManager.reasoning_parsers["muse_glimmer"] is MuseGlimmerReasoningParser
    assert ReasoningParserManager.get_reasoning_parser("muse_glimmer") is MuseGlimmerReasoningParser


def test_split_channels_finds_both_headers():
    assert [recipient for recipient, _ in split_channels(REASONED)] == ["self", "user"]
    assert [recipient for recipient, _ in split_channels(DIRECT)] == ["user"]
    assert split_channels("no header here at all") == []


def test_reasoned_turn_routes_analysis_away_from_content():
    reasoning, content = _parser().extract_reasoning(REASONED, request=None)
    assert reasoning == REASONED_ANALYSIS
    assert content == REASONED_REPLY
    # The constraint the unparsed string breaks and the parsed content keeps.
    assert content == content.lower()
    assert reasoning != reasoning.lower()


def test_full_special_token_framing_is_removed_from_plain_chat():
    reasoning, content = _parser().extract_reasoning(FULL_REASONED, request=None)
    assert reasoning == "Think first."
    assert content == "Done."
    assert "<|" not in reasoning + content


def test_direct_turn_has_no_reasoning():
    reasoning, content = _parser().extract_reasoning(DIRECT, request=None)
    assert reasoning is None
    assert content == DIRECT_REPLY


def test_unheadered_output_is_returned_unchanged():
    """A grammar-constrained or continuation generation must not be split."""
    raw = '{"city": "Paris", "population": 2148000}'
    reasoning, content = _parser().extract_reasoning(raw, request=None)
    assert reasoning is None
    assert content == raw


# Verbatim from the live server: `max_tokens=32`, temperature 0, the
# tt-inference-server coherence-guard prompt. `finish_reason` was `length` and
# the model was still restating the problem to itself
# (doc/tti_release/smoke/conformance_probe.json).
TRUNCATED_IN_ANALYSIS = (
    " to=selfRepeat the following sentence exactly, with no extra words, no"
    " quotes, and no commentary: The quick brown fox jumps over the lazy"
    " dog.\n\nWe"
)


def test_turn_cut_off_inside_the_analysis_channel_is_returned_unsplit():
    """The parser must never be able to empty `content`.

    A turn that runs out of `max_tokens` -- or hits a `stop` string -- inside
    the analysis channel has no visible channel.  Reporting `content=None` for
    it, which is what vLLM's `<think>`-style parsers do, throws away every token
    the model produced and hands `None` to clients that treat `content` as a
    string.  Such a turn is returned exactly as an unparsed server would return
    it, so enabling the parser can only ever *move* the analysis of a completed
    turn, never delete a response.
    """
    assert reached_visible_channel(REASONED) is True
    assert reached_visible_channel(DIRECT) is True
    assert reached_visible_channel(TRUNCATED_IN_ANALYSIS) is False

    reasoning, content = _parser().extract_reasoning(TRUNCATED_IN_ANALYSIS, request=None)
    assert reasoning is None
    assert content == TRUNCATED_IN_ANALYSIS
    # The property the coherence guard actually tests survives the parser.
    assert "The quick brown fox jumps over the lazy dog." in content


def test_content_is_always_a_string():
    for raw in (REASONED, DIRECT, TRUNCATED_IN_ANALYSIS, "", "plain text"):
        _, content = _parser().extract_reasoning(raw, request=None)
        assert isinstance(content, str), raw[:40]


def test_split_is_lossless_apart_from_the_headers():
    reasoning, content = _parser().extract_reasoning(REASONED, request=None)
    headers_stripped = REASONED.replace(" to=self", "", 1).replace("assistant to=user", "", 1)
    assert (reasoning or "") + (content or "") == headers_stripped


def test_composed_reasoning_and_tool_parsers_keep_calls_and_final_content():
    """Exercise vLLM's real DelegatingParser order, not either plugin alone."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager

    # Importing the plugin performs its vLLM registry hook.
    from models.autoports.meta_models_muse_glimmer_30b.tt import muse_glimmer_tool_parser  # noqa: F401

    req = ChatCompletionRequest(
        model="meta-models/Muse-Glimmer-30B",
        messages=[{"role": "user", "content": "inspect the file"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
        model_name=req.model,
    )
    assert parser_cls is not None
    parser = parser_cls(_FakeTokenizer(), tools=req.tools)
    parser.adjust_request(req)
    raw = (
        " to=self<|message|>I should inspect it.<|eom|>"
        "<|start|>assistant to=read_file<|message|>"
        '<atem:function_calls><atem:invoke name="read_file">'
        '<atem:parameter name="path">src/app.py</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eom|>"
        "<|start|>assistant to=user<|message|>Inspection requested.<|eot|>"
    )
    reasoning, content, calls = parser.parse(raw, req, enable_auto_tools=True)
    assert reasoning == "I should inspect it."
    assert content == "Inspection requested."
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == '{"path": "src/app.py"}'


def test_composed_parser_keeps_clean_content_when_tools_are_unused():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager

    from models.autoports.meta_models_muse_glimmer_30b.tt import muse_glimmer_tool_parser  # noqa: F401

    req = ChatCompletionRequest(
        model="meta-models/Muse-Glimmer-30B",
        messages=[{"role": "user", "content": "answer directly"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
        model_name=req.model,
    )
    assert parser_cls is not None
    parser = parser_cls(_FakeTokenizer(), tools=req.tools)
    parser.adjust_request(req)
    reasoning, content, calls = parser.parse(FULL_REASONED, req, enable_auto_tools=True)
    assert reasoning == "Think first."
    assert content == "Done."
    assert calls is None


def test_reasoning_end_is_true_when_no_analysis_channel_was_opened():
    """Structured output must get its grammar from the first step.

    ``is_reasoning_end`` gates when the grammar is applied.  A generation that
    never opens the analysis channel -- which is every grammar-constrained one,
    because the grammar is what stops it -- has to answer True immediately or
    the two would deadlock on each other.
    """
    parser = _parser(head_text="{")
    assert parser.is_reasoning_end([]) is True
    assert parser.is_reasoning_end([1, 2, 3]) is True


def test_reasoning_end_waits_for_eom_once_the_analysis_channel_is_open():
    parser = _parser(head_text=" to=self")
    assert parser.is_reasoning_end([1, 2, 3]) is False
    assert parser.is_reasoning_end([1, 2, 3, 200007]) is True


def test_delegating_parser_starts_new_assistant_turn_in_reasoning_phase():
    """Match vLLM's live prompt-to-stream lifecycle for a plain chat request.

    The rendered prompt ends at ``<|start|>assistant``; the generated channel
    header comes afterwards.  Treating the prompt as if reasoning had already
    ended bypasses this parser and leaks the raw self/user channel protocol to
    ``message.content``.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager

    from models.autoports.meta_models_muse_glimmer_30b.tt import muse_glimmer_tool_parser  # noqa: F401

    tokenizer = _LifecycleTokenizer()
    req = ChatCompletionRequest(
        model="meta-models/Muse-Glimmer-30B",
        messages=[{"role": "user", "content": "answer plainly"}],
        stream=True,
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
        model_name=req.model,
    )
    assert parser_cls is not None
    parser = parser_cls(tokenizer, tools=req.tools)
    parser.adjust_request(req)

    prompt_ids = [100, 200008, 200022, 140680]
    assert parser.is_reasoning_end(prompt_ids) is False

    generated_ids = [
        328,
        19669,
        200023,
        76221,
        200007,
        200022,
        140680,
        328,
        76976,
        200023,
        30550,
        200008,
    ]
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    previous_text = ""
    for i, token_id in enumerate(generated_ids):
        current_text = tokenizer.decode(generated_ids[: i + 1], skip_special_tokens=False)
        delta = parser.parse_delta(
            delta_text=current_text[len(previous_text) :],
            delta_token_ids=[token_id],
            request=req,
            prompt_token_ids=prompt_ids if i == 0 else None,
            finished=i == len(generated_ids) - 1,
        )
        previous_text = current_text
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)

    assert "".join(reasoning_parts) == "Think first."
    assert "".join(content_parts) == "Done."


def test_delegating_parser_keeps_a_direct_user_channel_visible():
    """A direct reply must survive the reasoning-to-content transition."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager

    from models.autoports.meta_models_muse_glimmer_30b.tt import muse_glimmer_tool_parser  # noqa: F401

    tokenizer = _LifecycleTokenizer()
    req = ChatCompletionRequest(
        model="meta-models/Muse-Glimmer-30B",
        messages=[{"role": "user", "content": "answer directly"}],
        stream=True,
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
        model_name=req.model,
    )
    assert parser_cls is not None
    parser = parser_cls(tokenizer, tools=req.tools)
    parser.adjust_request(req)

    prompt_ids = [100, 200008, 200022, 140680]
    generated_ids = [328, 76976, 200023, 30550, 200008]
    content_parts: list[str] = []
    previous_text = ""
    for i, token_id in enumerate(generated_ids):
        current_text = tokenizer.decode(generated_ids[: i + 1], skip_special_tokens=False)
        delta = parser.parse_delta(
            delta_text=current_text[len(previous_text) :],
            delta_token_ids=[token_id],
            request=req,
            prompt_token_ids=prompt_ids if i == 0 else None,
            finished=i == len(generated_ids) - 1,
        )
        previous_text = current_text
        if delta is not None and delta.content:
            content_parts.append(delta.content)

    assert "".join(content_parts) == "Done."


def test_delegating_parser_still_transitions_to_streamed_tool_calls():
    """The prompt-state fix must not strand active tools in reasoning."""
    import json

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager

    from models.autoports.meta_models_muse_glimmer_30b.tt import muse_glimmer_tool_parser  # noqa: F401

    tokenizer = _LifecycleTokenizer()
    req = ChatCompletionRequest(
        model="meta-models/Muse-Glimmer-30B",
        messages=[{"role": "user", "content": "run the probe"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record_latency_probe",
                    "description": "Record the probe",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "string"}},
                        "required": ["payload"],
                    },
                },
            }
        ],
        tool_choice="auto",
        stream=True,
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
        model_name=req.model,
    )
    assert parser_cls is not None
    parser = parser_cls(tokenizer, tools=req.tools)
    parser.adjust_request(req)

    prompt_ids = [100, 200008, 200022, 140680]
    tool_delta = (
        "<|start|>assistant to=record_latency_probe<|message|>"
        '<atem:function_calls><atem:invoke name="record_latency_probe">'
        '<atem:parameter name="payload">ready</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )
    chunks = [
        (" to", [328]),
        ("=self", [19669]),
        ("<|message|>", [200023]),
        ("Think first.", [76221]),
        ("<|eom|>", [200007]),
        (tool_delta, [999]),
    ]
    calls = []
    reasoning_parts: list[str] = []
    for i, (delta_text, delta_ids) in enumerate(chunks):
        delta = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=delta_ids,
            request=req,
            prompt_token_ids=prompt_ids if i == 0 else None,
            finished=i == len(chunks) - 1,
        )
        if delta is None:
            continue
        reasoning_parts.append(delta.reasoning or "")
        calls.extend(delta.tool_calls or [])

    assert "".join(reasoning_parts) == "Think first."
    assert len(calls) == 1
    assert calls[0].function.name == "record_latency_probe"
    assert json.loads(calls[0].function.arguments) == {"payload": "ready"}


def test_content_ids_start_after_the_analysis_channel():
    parser = _parser(head_text=" to=self")
    assert parser.extract_content_ids([1, 2, 200007, 7, 8]) == [7, 8]
    assert parser.count_reasoning_tokens([1, 2, 200007, 7, 8]) == 3
    direct = _parser(head_text=" to=user")
    assert direct.extract_content_ids([1, 2, 3]) == [1, 2, 3]
    assert direct.count_reasoning_tokens([1, 2, 3]) == 0


@pytest.mark.parametrize("chunk", [1, 7, 40])
def test_streaming_deltas_reassemble_the_same_split(chunk):
    """Streaming must land every character in the same channel as one-shot."""
    parser = _parser()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for i in range(0, len(REASONED), chunk):
        previous_text = REASONED[:i]
        current_text = REASONED[: i + chunk]
        delta = parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            current_text[len(previous_text) :],
            [],
            [],
            [],
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
    assert "".join(reasoning_parts) == REASONED_ANALYSIS
    assert "".join(content_parts) == REASONED_REPLY
