# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host tests for ``tt/tokenizer_chat.py`` against transformers' own tokenizer + chat template of the checkpoint.

Needs the tokenizer files of a GLM-5.3-Flash snapshot ($GLM53_HF_SNAPSHOT, else the local HF hub cache)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.experimental.glm53_flash.tt.tokenizer_chat import (
    EXPECTED_STOP_TOKENS,
    MODEL_VOCAB_SIZE,
    TOKENIZER_FILES,
    load_tokenizer,
    resolve_snapshot,
)


def _find_snapshot() -> Path | None:
    roots = [os.environ.get("GLM53_HF_SNAPSHOT")]
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--zai-org--GLM-5.3-Flash" / "snapshots"
    if hub.is_dir():
        roots += [str(p) for p in sorted(hub.iterdir())]
    for r in roots:
        if r and all((Path(r) / f).exists() for f in TOKENIZER_FILES):
            return Path(r)
    return None


SNAPSHOT = _find_snapshot()
pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="no GLM-5.3-Flash tokenizer snapshot")


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(SNAPSHOT)


@pytest.fixture(scope="module")
def hf_tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(SNAPSHOT))


TEXTS = [
    "Hello, world!",
    "  leading and trailing spaces  ",
    "多语言 テキスト 한국어 — emoji 🤖🚀",
    "def f(x):\n    return x ** 2  # comment\n\n\ttab",
    "<|user|> special tokens inside plain text <think></think>",
    "",
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}

CONVERSATIONS = {
    "single_user": [{"role": "user", "content": "Hi there"}],
    "system_multi_turn": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple addition."},
        {"role": "user", "content": "And 3+3?"},
    ],
    "tool_round_trip": [
        {"role": "user", "content": "Weather in Paris for 2 days?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}
            ],
        },
        {"role": "tool", "content": '{"temp_c": 18}'},
    ],
}


# ---------------------------------------------------------------------------------------------
# Loading and provenance
# ---------------------------------------------------------------------------------------------


def test_resolve_snapshot_explicit_and_missing(tmp_path, expect_error):
    assert resolve_snapshot(SNAPSHOT) == SNAPSHOT
    with expect_error(FileNotFoundError, ""):
        resolve_snapshot(tmp_path)


def test_stop_tokens_and_vocab(tok):
    assert {i: tok.id_to_token(i) for i in tok.stop_token_ids} == EXPECTED_STOP_TOKENS
    assert all(tok.is_stop_token(i) for i in EXPECTED_STOP_TOKENS) and not tok.is_stop_token(0)
    assert tok.eos_token_id in tok.stop_token_ids
    assert tok.vocab_size <= MODEL_VOCAB_SIZE == tok.model_vocab_size
    d = tok.describe()
    assert d["stop_token_ids"] == list(tok.stop_token_ids) and set(d["files_sha256"]) == set(TOKENIZER_FILES)


def test_matches_transformers_vocab(hf_tok, tok):
    assert len(hf_tok) == tok.vocab_size
    assert hf_tok.eos_token_id == tok.eos_token_id and hf_tok.pad_token_id == tok.pad_token_id


# ---------------------------------------------------------------------------------------------
# Token level
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
def test_encode_decode_match_transformers(hf_tok, tok, text):
    ids = tok.encode(text)
    assert ids == hf_tok.encode(text, add_special_tokens=False) == hf_tok.encode(text, add_special_tokens=True)
    assert tok(text)["input_ids"] == ids
    assert tok.decode(ids) == hf_tok.decode(ids) == text
    assert tok.decode(ids, skip_special_tokens=True) == hf_tok.decode(ids, skip_special_tokens=True)
    assert tok.convert_ids_to_tokens(ids) == hf_tok.convert_ids_to_tokens(ids)


def test_decode_rejects_padding_ids(tok, expect_error):
    with expect_error(ValueError, "outside the tokenizer vocabulary"):
        tok.decode([1, tok.vocab_size])
    with expect_error(KeyError, ""):
        tok.token_to_id("<definitely-not-a-token>")


# ---------------------------------------------------------------------------------------------
# Chat level
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CONVERSATIONS))
@pytest.mark.parametrize("effort", [None, "low", "high"])
def test_render_chat_matches_transformers(hf_tok, tok, name, effort):
    msgs = CONVERSATIONS[name]
    tools = [WEATHER_TOOL] if name == "tool_round_trip" else None
    kw = {} if effort is None else {"reasoning_effort": effort}
    ours = tok.render_chat(msgs, tools=tools, **kw)
    ref = hf_tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw)
    assert ours == ref
    assert (
        tok.apply_chat_template(msgs, tools=tools, **kw) == tok.encode(ref) == tok.encode_chat(msgs, tools=tools, **kw)
    )
    assert ours.endswith("<|assistant|><think>")


def test_clear_thinking_matches_transformers(hf_tok, tok):
    msgs = CONVERSATIONS["system_multi_turn"]
    ours = tok.render_chat(msgs, clear_thinking=True)
    assert ours == hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, clear_thinking=True)
    assert "Simple addition." not in ours and "Simple addition." in tok.render_chat(msgs)


def test_continue_final_message_matches_transformers(hf_tok, tok):
    msgs = [{"role": "user", "content": "Count to three."}, {"role": "assistant", "content": "One, two,"}]
    ours = tok.render_chat(msgs, add_generation_prompt=False, continue_final_message=True)
    ref = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    assert ours == ref and ours.endswith("One, two,")


def test_render_chat_validation(tok, expect_error):
    with expect_error(ValueError, "empty"):
        tok.render_chat([])
    with expect_error(ValueError, "not compatible"):
        tok.render_chat(CONVERSATIONS["single_user"], continue_final_message=True)
    with expect_error(TypeError, ""):
        tok.render_chat(CONVERSATIONS["single_user"], tools=[lambda city: city])


# ---------------------------------------------------------------------------------------------
# Output side
# ---------------------------------------------------------------------------------------------


def test_parse_assistant_text(tok):
    p = tok.parse_assistant("Let me think.</think>The answer is 4.")
    assert (p.reasoning_content, p.content, p.tool_calls) == ("Let me think.", "The answer is 4.", [])
    assert tok.parse_assistant("No reasoning block").reasoning_content is None


def test_parse_assistant_ids_round_trip_tool_call(tok):
    """Render a tool call the way the template writes it back, encode it, append a stop token, parse the ids."""
    hist = CONVERSATIONS["tool_round_trip"][:2]
    rendered = tok.render_chat(hist, tools=[WEATHER_TOOL], add_generation_prompt=False)
    tail = rendered.split("<|assistant|>")[-1]
    ids = tok.encode("Checking.</think>" + tail.split("</think>", 1)[-1]) + [tok.eos_token_id]
    p = tok.parse_assistant(ids)
    assert p.reasoning_content == "Checking."
    assert p.tool_calls == [{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}]
