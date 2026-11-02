# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: packed-forward metadata codec, the order contract and the packed H2D token layout."""

import struct

import pytest
import torch

from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotate_chunk_tokens
from models.demos.common.prefill.prefill_chunk import (
    EMPTY_CHUNK,
    INVALID_SLOT,
    SHUTDOWN_WORD,
    WARMUP_WORD,
    PrefillChunk,
    decode_forward_metadata,
    forward_metadata_words,
    metadata_size_bytes,
    pack_forward_metadata,
    packed_forward_refusals,
    pad_chunks,
    sentinel_metadata,
    validate_chunks,
)

C, SP = 1024, 8
S = C // SP
EMPTY = EMPTY_CHUNK


def as_int32_words(metadata: bytes) -> list:
    return list(struct.unpack(f"<{len(metadata) // 4}i", metadata))


def test_metadata_size_bytes(expect_error):
    assert [metadata_size_bytes(f) for f in (1, 2, 4, 16, 32)] == [12, 32, 56, 200, 392]
    with expect_error(ValueError, "a forward of 0 chunks"):
        metadata_size_bytes(0)


def test_batched_layout_is_prefill_batch_metadata():
    chunks = [PrefillChunk(0, 0, C), PrefillChunk(1, C, 2 * C), PrefillChunk(1, 2 * C, 2500), EMPTY]
    expected = struct.pack("<11I", 3, 0, 0, 0, C, 1, C, 2 * C, 1, 2 * C, 2500)
    expected += struct.pack("<3I", INVALID_SLOT, 0, 0)
    assert pack_forward_metadata(chunks, 4) == expected
    assert decode_forward_metadata(as_int32_words(expected), 4) == chunks


def test_batched_layout_rejects_bad_counts(expect_error):
    words = as_int32_words(pack_forward_metadata([PrefillChunk(0, 0, C), EMPTY], 2))
    for n in (0, 3):
        with expect_error(ValueError, f"num_chunks={n} outside"):
            decode_forward_metadata([n] + words[1:], 2)
    with expect_error(ValueError, "empty chunks must come after"):
        pack_forward_metadata([EMPTY, PrefillChunk(0, 0, C)], 2)
    with expect_error(ValueError, "do not fit a forward of 2"):
        pack_forward_metadata([PrefillChunk(0, 0, C)] * 3, 2)


def test_batched_decode_rejects_a_single_chunk_record(expect_error):
    words = as_int32_words(pack_forward_metadata([PrefillChunk(0, 0, C), EMPTY], 2))
    with expect_error(ValueError, "reserved header word is 5"):
        decode_forward_metadata(words[:1] + [5] + words[2:], 2)
    with expect_error(ValueError, "record 1 past num_chunks=1"):
        decode_forward_metadata(words[:5] + [2, 0, 0], 2)
    # The 12 B record of a producer still at one chunk per forward, read by a runner at four.
    one_chunk = as_int32_words(pack_forward_metadata([PrefillChunk(3, C, 2 * C)], 1))
    with expect_error(ValueError, "outside|reserved|past num_chunks"):
        decode_forward_metadata(one_chunk + [0] * 11, 4)


def test_encoder_rejects_an_empty_forward(expect_error):
    with expect_error(ValueError, "at least one chunk"):
        forward_metadata_words([], 2)
    with expect_error(ValueError, "only empty chunks"):
        forward_metadata_words([EMPTY, EMPTY], 2)


def test_single_chunk_is_todays_metadata():
    assert pack_forward_metadata([PrefillChunk(3, 2048, 2900)], 1) == struct.pack("<3I", 3, 2048, 2900)
    assert sentinel_metadata(SHUTDOWN_WORD, 1) == struct.pack("<iii", -1, -1, -1)


def test_unpadded_chunk_takes_the_forward_layout():
    chunk = PrefillChunk(3, 2048, 2900)
    metadata = pack_forward_metadata([chunk], 4)
    assert len(metadata) == metadata_size_bytes(4)
    assert decode_forward_metadata(as_int32_words(metadata), 4) == pad_chunks([chunk], 4)


@pytest.mark.parametrize("forward_chunks", range(1, 9))
def test_roundtrip(forward_chunks):
    gen = torch.Generator().manual_seed(forward_chunks)

    def randint(lo, hi):  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    for _ in range(50):
        chunks = []
        for _ in range(randint(1, forward_chunks)):
            start = 32 * randint(0, (1 << 20) // 32 - 1)
            chunks.append(PrefillChunk(randint(0, 63), start, start + randint(1, C)))
        chunks = pad_chunks(chunks, forward_chunks)
        metadata = pack_forward_metadata(chunks, forward_chunks)
        assert len(metadata) == metadata_size_bytes(forward_chunks)
        assert decode_forward_metadata(as_int32_words(metadata), forward_chunks) == chunks


@pytest.mark.parametrize("forward_chunks", [1, 3, 8, 16, 32])
@pytest.mark.parametrize("word", [SHUTDOWN_WORD, WARMUP_WORD])
def test_sentinels_fill_every_word(word, forward_chunks):
    metadata = sentinel_metadata(word, forward_chunks)
    assert len(metadata) == metadata_size_bytes(forward_chunks)
    assert as_int32_words(metadata) == [word] * (metadata_size_bytes(forward_chunks) // 4)
    chunks = decode_forward_metadata(as_int32_words(metadata), forward_chunks)
    assert chunks == [PrefillChunk(word, word, word)] * forward_chunks
    # A rank forwards the sentinel it decoded over D2D.
    assert forward_metadata_words(chunks, forward_chunks) == as_int32_words(metadata)


def test_decode_ignores_trailing_words_and_rejects_short(expect_error):
    words = as_int32_words(pack_forward_metadata([PrefillChunk(1, 0, C), PrefillChunk(2, 0, 7)], 2)) + [5]
    assert decode_forward_metadata(words, 2) == [PrefillChunk(1, 0, C), PrefillChunk(2, 0, 7)]
    with expect_error(ValueError, "needs 11"):
        decode_forward_metadata(words[:6], 3)


def test_pad_chunks(expect_error):
    assert pad_chunks([PrefillChunk(1, 0, C)], 3) == [PrefillChunk(1, 0, C), EMPTY, EMPTY]
    with expect_error(ValueError, "do not fit"):
        pad_chunks([PrefillChunk(1, 0, C)] * 3, 2)


GOOD = {
    "single": [PrefillChunk(0, 0, C)],
    "all_full": [
        PrefillChunk(0, 0, C),
        PrefillChunk(0, C, 2 * C),
        PrefillChunk(1, 5 * C, 6 * C),
        PrefillChunk(2, 0, C),
    ],
    "partial_last": [
        PrefillChunk(0, 0, C),
        PrefillChunk(1, 0, C),
        PrefillChunk(2, 2 * C, 3 * C),
        PrefillChunk(3, 0, 300),
    ],
    # A partial-ending run may precede another slot's full run: the order across slots is free.
    "partial_before_full": [
        PrefillChunk(0, 0, C),
        PrefillChunk(0, C, C + 700),
        PrefillChunk(1, 4 * C, 5 * C),
        PrefillChunk(2, 0, 300),
    ],
    "partial_first": [PrefillChunk(2, 0, 300), PrefillChunk(0, 0, C), PrefillChunk(1, 0, C), EMPTY],
    "empty_tail": [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, 0, 600), EMPTY, EMPTY],
    "mid_slab": [
        PrefillChunk(0, 2 * C + 96, 3 * C + 96),
        PrefillChunk(0, 3 * C + 96, 3 * C + 900),
        PrefillChunk(1, C + 512, 2 * C + 512),
        EMPTY,
    ],
    "empty_any_slot": [PrefillChunk(0, 0, C), PrefillChunk(9, 64, 64), EMPTY, EMPTY],
}


@pytest.mark.parametrize("name", GOOD)
def test_validate_accepts(name):
    chunks = GOOD[name]
    validate_chunks(chunks, chunk_size=C, forward_chunks=len(chunks), num_users=4, max_seq_len=8 * C)


BAD = {  # name -> (chunks, expected message)
    "wrong_count": ([PrefillChunk(0, 0, C)] * 3, "holds 4 chunks"),
    "split_run": (
        [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C), PrefillChunk(0, C, 2 * C), EMPTY],
        "two separate runs",
    ),
    "partial_not_last": ([PrefillChunk(0, 0, 500), PrefillChunk(0, C, 2 * C), EMPTY, EMPTY], "must end its slot"),
    "gap_in_run": ([PrefillChunk(0, 0, C), PrefillChunk(0, 2 * C, 3 * C), EMPTY, EMPTY], "does not continue"),
    "repeated_chunk": (
        [PrefillChunk(0, 6 * C, 7 * C), PrefillChunk(0, 7 * C, 8 * C), PrefillChunk(0, 7 * C, 8 * C), EMPTY],
        "does not continue",
    ),
    # tenstorrent/tt-d-gen#430's clamp: a last chunk moved back to end at the cache end.
    "overlap_at_cache_end": (
        [PrefillChunk(0, 6 * C + 96, 7 * C + 96), PrefillChunk(0, 7 * C, 7 * C + 700), EMPTY, EMPTY],
        "does not continue",
    ),
    "overlap_not_at_cache_end": (
        [PrefillChunk(0, C + 96, 2 * C + 96), PrefillChunk(0, 2 * C, 3 * C), EMPTY, EMPTY],
        "does not continue",
    ),
    "empty_not_tail": ([PrefillChunk(0, 0, C), EMPTY, PrefillChunk(1, 0, C), EMPTY], "after an empty one"),
    "all_empty": ([EMPTY] * 4, "at least one non-empty chunk"),
    "negative_start": ([PrefillChunk(0, -32, 32), EMPTY, EMPTY, EMPTY], "negative start"),
    "misaligned_start": ([PrefillChunk(0, 100, 600), EMPTY, EMPTY, EMPTY], "multiple of 32"),
    "more_than_a_chunk": ([PrefillChunk(0, 0, C + 32), EMPTY, EMPTY, EMPTY], "more than one"),
    "end_before_start": ([PrefillChunk(0, 64, 32), EMPTY, EMPTY, EMPTY], "more than one"),
    "cache_overflow": ([PrefillChunk(0, 7 * C + 32, 7 * C + 64), EMPTY, EMPTY, EMPTY], "exceeds the per-user cache"),
    "slot_out_of_range": ([PrefillChunk(4, 0, C), EMPTY, EMPTY, EMPTY], "slot out of range"),
    "shutdown_sentinel": ([PrefillChunk(-1, -1, -1)] * 4, "at least one non-empty chunk"),
}


@pytest.mark.parametrize("name", BAD)
def test_validate_rejects(name, expect_error):
    chunks, message = BAD[name]
    with expect_error(ValueError, message):
        validate_chunks(chunks, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)


@pytest.mark.parametrize("starts", [[0], [0, C, 3 * C, 0], [2 * C + 96, 3 * C + 96, C + 512, 0]])
def test_pack_forward_tokens(starts):
    chunks = [[1000 * k + i for i in range(C)] for k in range(len(starts))]
    packed = pack_forward_tokens(chunks, starts, SP)
    width = len(starts) * S
    for c in range(SP):
        chip = packed[c * width : (c + 1) * width]
        for k, (tokens, start) in enumerate(zip(chunks, starts)):
            assert chip[k * S : (k + 1) * S] == rotate_chunk_tokens(tokens, start, SP)[c * S : (c + 1) * S]
    if len(starts) == 1:
        assert packed == rotate_chunk_tokens(chunks[0], starts[0], SP)


def test_packed_forward_refusals():
    assert packed_forward_refusals(1, supports_packed_forward=False, mtp_levels=2, use_trace=True) == []
    assert packed_forward_refusals(4, supports_packed_forward=True) == []
    reasons = packed_forward_refusals(
        4,
        supports_packed_forward=False,
        mtp_levels=1,
        dflash=True,
        use_trace=True,
        d2h_acks=True,
        indexer_indices=True,
    )
    assert len(reasons) == 6
    for knob in ("supports_packed_forward", "MTP", "DFLASH", "USE_TRACE", "LAYER_ACK_D2H", "indexer indices"):
        assert any(knob in r for r in reasons), knob
