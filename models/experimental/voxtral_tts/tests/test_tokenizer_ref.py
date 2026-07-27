# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tekken tokenizer + prompt-assembly tests.

The gate that matters is EXACT TOKEN-ID EQUALITY with `mistral_common`. `prompt_fixture.json`
vendors 15 ground-truth prompts produced by mistral_common 1.11.7's `encode_speech_request`
(English, French, German, Spanish, Italian, Portuguese, Hindi, Arabic, digits, symbols, emoji,
tabs/newlines, and a 125-word paragraph, across 10 different voices), so these tests run without
mistral_common installed. Regenerate with scripts/dump_prompt_ids.py-style code if the upstream
template ever changes — a diff here means our prompt no longer matches the real one.

Needs tekken.json (14 MB, downloaded with the checkpoint); skips cleanly without it.

    pytest -svv models/experimental/voxtral_tts/tests/test_tokenizer_ref.py
"""

import json
import os

import pytest

from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import (
    AUDIO,
    BEGIN_AUDIO,
    BOS,
    DEFAULT_TEKKEN,
    NEXT_AUDIO_TEXT,
    REPEAT_AUDIO_TEXT,
    TekkenTokenizer,
    _bpe,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "prompt_fixture.json")
needs_tekken = pytest.mark.skipif(not os.path.exists(DEFAULT_TEKKEN), reason=f"no tekken.json at {DEFAULT_TEKKEN}")
pytestmark = needs_tekken


@pytest.fixture(scope="module")
def tok():
    return TekkenTokenizer()


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def _cases():
    """Parametrize ids at collection time so each case is a named test."""
    if not os.path.exists(FIXTURE):
        return []
    with open(FIXTURE) as f:
        d = json.load(f)
    return [pytest.param(c, id=f"{c['voice']}-{len(c['ids'])}ids-{c['text'][:14].strip()!r}") for c in d["cases"]]


# ---------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", _cases())
def test_prompt_ids_match_mistral_common_exactly(tok, case):
    ours = tok.build_prompt(case["text"], case["voice"])
    theirs = case["ids"]
    assert len(ours) == len(theirs), f"length {len(ours)} vs {len(theirs)}"
    if ours != theirs:
        i = next(k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b)
        pytest.fail(f"first divergence at index {i}: ours {ours[i]} vs mistral_common {theirs[i]}")


def test_fixture_covers_non_latin_and_long_text(fixture):
    """Guards the fixture itself: an all-ASCII fixture would pass even with a broken Unicode
    split pattern, which is the most likely way this reimplementation goes wrong."""
    texts = [c["text"] for c in fixture["cases"]]
    assert any(not t.isascii() for t in texts), "fixture has no non-ASCII case"
    assert any(len(t.split()) > 100 for t in texts), "fixture has no long-text case"
    assert len({c["voice"] for c in fixture["cases"]}) >= 8, "fixture should span many voices"


def test_audio_token_id_agrees_with_fixture(tok, fixture):
    assert tok.audio_token_id == fixture["audio_token_id"] == 24


# ---------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------
def test_prompt_layout(tok):
    """<s> [BEGIN_AUDIO] [AUDIO]xN [NEXT_AUDIO_TEXT] <text> [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]"""
    sp = tok.special
    ids = tok.build_prompt("Hello there.", "neutral_male")
    n = tok.n_audio_tokens("neutral_male")
    assert ids[0] == sp[BOS]
    assert ids[1] == sp[BEGIN_AUDIO]
    assert ids[2 : 2 + n] == [sp[AUDIO]] * n
    assert ids[2 + n] == sp[NEXT_AUDIO_TEXT]
    assert ids[-2] == sp[REPEAT_AUDIO_TEXT]
    assert ids[-1] == sp[BEGIN_AUDIO]
    assert tok.decode(ids[3 + n : -2]) == "Hello there."


def test_placeholder_count_is_voice_specific(tok):
    """The reason a prompt cannot be reused across voices. Values come from tekken.json's
    audio.voice_num_audio_tokens, so no preset file is needed to know them."""
    counts = {v: tok.n_audio_tokens(v) for v in tok.voices}
    assert len(counts) == 20
    assert counts["neutral_male"] == 169 and counts["cheerful_female"] == 132
    assert counts["ar_male"] == 67 == min(counts.values())
    assert counts["neutral_female"] == 218 == max(counts.values())
    assert len(set(counts.values())) > 10, "counts should genuinely vary per voice"


def test_placeholder_counts_match_preset_row_counts(tok):
    """tekken.json's per-voice frame count must equal the shipped preset's row count, or the
    pipeline's substitution asserts. Skips per-voice if a preset was not downloaded."""
    import torch

    from models.experimental.voxtral_tts.reference.voxtral_pipeline_ref import VOICE_DIR

    if not os.path.isdir(VOICE_DIR):
        pytest.skip("voice_embedding/ not downloaded")
    checked = 0
    for v in tok.voices:
        p = os.path.join(VOICE_DIR, f"{v}.pt")
        if not os.path.exists(p):
            continue
        rows = torch.load(p, map_location="cpu", weights_only=False).shape[0]
        assert rows == tok.n_audio_tokens(v), f"{v}: preset {rows} rows vs tekken {tok.n_audio_tokens(v)}"
        checked += 1
    assert checked >= 1, "no presets found to check"


def test_unknown_voice_raises(tok):
    with pytest.raises(KeyError, match="unknown voice"):
        tok.build_prompt("hi", "no_such_voice")


# ---------------------------------------------------------------------------------------
# BPE mechanics
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    ["It took me quite a long time.", "Café déjà vu", "1234567890", "!@#$%^&*()",
     "emoji 🎤🔊 test", "  leading and   inner spaces", "Tab\tand\nnewline", "नमस्ते", "مرحبا",
     "CAPS lower MiXeD", ""],
)
def test_encode_decode_round_trip(tok, text):
    """Byte-level BPE must be lossless, including for multi-byte codepoints."""
    assert tok.decode(tok.encode(text)) == text


def test_ids_are_in_range(tok):
    ids = tok.encode("The quick brown fox — jumps! 42 times. नमस्ते 🎤")
    assert all(tok.n_special <= i < tok.vocab_size for i in ids), "regular ids must be offset past specials"


def test_bpe_prefers_lowest_rank_merge():
    """The merge order IS the algorithm: always merge the lowest-ranked adjacent pair. A
    left-to-right or highest-first variant produces different, wrong ids."""
    ranks = {b"a": 0, b"b": 1, b"c": 2, b"bc": 3, b"ab": 9}
    # 'abc': merging 'bc' (rank 3) beats 'ab' (rank 9), leaving ['a', 'bc']
    assert _bpe(ranks, b"abc") == [0, 3]
    ranks2 = {b"a": 0, b"b": 1, b"c": 2, b"bc": 9, b"ab": 3}
    assert _bpe(ranks2, b"abc") == [3, 2]


def test_bpe_returns_whole_piece_when_present():
    ranks = {b"a": 0, b"b": 1, b"ab": 5}
    assert _bpe(ranks, b"ab") == [5]


def test_vocab_truncated_to_released_size(tok):
    """tekken.json ships 150000 vocab entries but the embedding table is 131072 wide; ids past
    that would index out of bounds in tok_embeddings."""
    assert tok.vocab_size == 131072
    assert len(tok.ranks) == 131072 - 1000
    assert max(tok.by_rank) == 131072 - 1000 - 1


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
