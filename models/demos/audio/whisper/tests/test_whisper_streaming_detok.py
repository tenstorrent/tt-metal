# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for Whisper streaming incremental detokenization.

These exercise ``WhisperGenerator._stream_incremental_texts`` in isolation — no device,
no model weights, just the HF Whisper tokenizer. They guard the subtle windowed UTF-8
logic that fixes multi-byte (CJK) garbling, asserting that:

  1. The concatenation of streaming incremental yields equals the single full-sequence
     decode (so streaming and non-streaming agree) for multi-byte and mixed text.
  2. No spurious U+FFFD replacement characters leak into the streamed text mid-stream.
  3. The hold never stalls: a token that can never complete a character is flushed once
     the cache reaches STREAM_MAX_HOLD_TOKENS.
  4. Batching multiple batch elements through one call keeps each element independent.
"""

import pytest
import transformers

from models.demos.audio.whisper.tt.whisper_generator import STREAM_MAX_HOLD_TOKENS, WhisperGenerator

MODEL_NAME = "distil-whisper/distil-large-v3"

# Multi-byte (CJK) and mixed-script strings whose byte-level BPE splits characters across tokens.
MULTI_BYTE_TEXTS = [
    "こんにちは世界",  # Japanese
    "日本語の音声認識テストです",  # Japanese, longer
    "你好世界",  # Chinese
    "안녕하세요 세계",  # Korean
    "Hello 世界 mixed 漢字 text 123",  # mixed ASCII + CJK
    "絵文字 😀 と日本語",  # 4-byte UTF-8 (emoji) + CJK
]


@pytest.fixture(scope="module")
def tokenizer():
    # The tokenizer alone is light to download relative to the full model and runs on CPU.
    # _stream_incremental_texts only needs an object exposing batch_decode(); the tokenizer
    # provides it (the production processor delegates batch_decode to this same tokenizer).
    return transformers.WhisperTokenizer.from_pretrained(MODEL_NAME)


def _stream_full(tokenizer, ids):
    """Feed ``ids`` one token at a time through a single-element streaming cache and return
    the concatenation of the incremental yields."""
    caches = [[]]
    streamed = ""
    for tid in ids:
        streamed += WhisperGenerator._stream_incremental_texts(tokenizer, caches, [tid])[0]
    return streamed


@pytest.mark.parametrize("text", MULTI_BYTE_TEXTS)
def test_streaming_concatenation_matches_full_decode(tokenizer, text):
    """Streaming incremental yields must concatenate to the same string as the one-shot decode."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    full = tokenizer.batch_decode([ids], skip_special_tokens=True)[0]

    streamed = _stream_full(tokenizer, ids)

    assert streamed == full, f"streamed {streamed!r} != full decode {full!r}"
    # A complete, valid UTF-8 sequence must never leave a replacement char in the stream.
    assert "�" not in streamed


def test_per_token_naive_decode_garbles_but_windowed_does_not(tokenizer):
    """Sanity: the naive per-token decode (the original bug) garbles CJK; the windowed path fixes it.

    Not every CJK character splits across tokens, so search the candidates for one that the naive
    per-token decode actually garbles, then assert the windowed path reconstructs it exactly.
    """
    garbled_example = None
    for text in MULTI_BYTE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        naive = "".join(tokenizer.batch_decode([[tid] for tid in ids], skip_special_tokens=True))
        if "�" in naive:
            garbled_example = (text, ids)
            break

    assert garbled_example is not None, "expected at least one candidate to garble under naive per-token decode"
    text, ids = garbled_example
    assert _stream_full(tokenizer, ids) == text


def test_hold_never_stalls_on_undecodable_byte(tokenizer):
    """A lone continuation byte can never complete a character; it must flush at the hold cap."""
    # 0x80 is a lone UTF-8 continuation byte. In Whisper byte-level BPE its single-byte token
    # is the byte-fallback token for 0x80, which decodes to U+FFFD in isolation forever.
    cont_id = tokenizer.convert_tokens_to_ids("ĉ")  # GPT-2 byte map: 0x80 -> 'ĉ'
    assert cont_id is not None and cont_id != tokenizer.unk_token_id

    caches = [[]]
    yields = []
    # Feed more than the cap so the hold is forced to release.
    for _ in range(STREAM_MAX_HOLD_TOKENS + 2):
        yields.append(WhisperGenerator._stream_incremental_texts(tokenizer, caches, [cont_id])[0])

    # Held (empty yields) until the cap, then flushed (non-empty) — the stream must not stall forever.
    assert any(y != "" for y in yields), "undecodable byte was held forever — stream stalled"
    # After a flush the cache is cleared so it does not grow unbounded.
    assert len(caches[0]) <= STREAM_MAX_HOLD_TOKENS


def test_batched_elements_are_independent(tokenizer):
    """Decoding multiple batch elements in one call must keep each element's stream independent."""
    texts = ["こんにちは", "Hello world", "你好"]
    id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    max_len = max(len(ids) for ids in id_lists)

    caches = [[] for _ in texts]
    streamed = ["" for _ in texts]
    for step in range(max_len):
        # Pad shorter sequences with EOS so all elements step together (extra EOS decodes to "").
        step_ids = [ids[step] if step < len(ids) else tokenizer.eos_token_id for ids in id_lists]
        pieces = WhisperGenerator._stream_incremental_texts(tokenizer, caches, step_ids)
        for b in range(len(texts)):
            streamed[b] += pieces[b]

    for b, text in enumerate(texts):
        assert streamed[b] == text, f"element {b}: {streamed[b]!r} != {text!r}"


def test_token_ids_longer_than_caches_are_ignored(tokenizer):
    """Padded batch: token_ids longer than caches must not raise and must ignore the padding."""
    ids = tokenizer.encode("世界", add_special_tokens=False)
    caches = [[]]
    streamed = ""
    for tid in ids:
        # One real cache, two token ids (a padded batch entry) — the extra id must be ignored.
        streamed += WhisperGenerator._stream_incremental_texts(tokenizer, caches, [tid, tokenizer.eos_token_id])[0]
    assert streamed == "世界"
