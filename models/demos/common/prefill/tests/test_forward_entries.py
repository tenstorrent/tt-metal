# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: packed-forward metadata codec, the order contract and the packed H2D token layout."""

import random
import struct

import pytest

from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotate_chunk_tokens
from models.demos.common.prefill.forward_entries import (
    EMPTY_ENTRY,
    SHUTDOWN_WORD,
    WARMUP_WORD,
    PrefillEntry,
    decode_forward_metadata,
    metadata_size_bytes,
    pack_forward_metadata,
    packed_forward_refusals,
    pad_entries,
    sentinel_metadata,
    validate_forward_entries,
)

C, SP = 1024, 8
S = C // SP
EMPTY = EMPTY_ENTRY


def as_int32_words(metadata: bytes) -> list:
    return list(struct.unpack(f"<{len(metadata) // 4}i", metadata))


def test_metadata_size_bytes():
    assert [metadata_size_bytes(f) for f in (1, 2, 4, 8)] == [12, 24, 48, 96]


def test_single_entry_is_todays_metadata():
    assert pack_forward_metadata([PrefillEntry(3, 2048, 2900)]) == struct.pack("<3I", 3, 2048, 2900)
    assert sentinel_metadata(SHUTDOWN_WORD, 1) == struct.pack("<iii", -1, -1, -1)


@pytest.mark.parametrize("forward_chunks", range(1, 9))
def test_roundtrip(forward_chunks):
    rng = random.Random(forward_chunks)
    for _ in range(50):
        n = rng.randint(1, forward_chunks)
        entries = []
        for _ in range(n):
            start = rng.randrange(0, 1 << 20, 32)
            entries.append(PrefillEntry(rng.randrange(64), start, start + rng.randint(1, C)))
        entries = pad_entries(entries, forward_chunks)
        metadata = pack_forward_metadata(entries)
        assert len(metadata) == metadata_size_bytes(forward_chunks)
        assert decode_forward_metadata(as_int32_words(metadata), forward_chunks) == entries


@pytest.mark.parametrize("forward_chunks", [1, 3, 8])
@pytest.mark.parametrize("word", [SHUTDOWN_WORD, WARMUP_WORD])
def test_sentinels_fill_every_word(word, forward_chunks):
    metadata = sentinel_metadata(word, forward_chunks)
    assert len(metadata) == metadata_size_bytes(forward_chunks)
    assert as_int32_words(metadata) == [word] * (3 * forward_chunks)
    entries = decode_forward_metadata(as_int32_words(metadata), forward_chunks)
    assert entries == [PrefillEntry(word, word, word)] * forward_chunks


def test_decode_ignores_trailing_words_and_rejects_short(expect_error):
    words = as_int32_words(pack_forward_metadata([PrefillEntry(1, 0, C), PrefillEntry(2, 0, 7)])) + [5]
    assert decode_forward_metadata(words, 2) == [PrefillEntry(1, 0, C), PrefillEntry(2, 0, 7)]
    with expect_error(ValueError, "needs 9"):
        decode_forward_metadata(words[:6], 3)


def test_pad_entries(expect_error):
    assert pad_entries([PrefillEntry(1, 0, C)], 3) == [PrefillEntry(1, 0, C), EMPTY, EMPTY]
    with expect_error(ValueError, "do not fit"):
        pad_entries([PrefillEntry(1, 0, C)] * 3, 2)


GOOD = {
    "single": [PrefillEntry(0, 0, C)],
    "all_full": [
        PrefillEntry(0, 0, C),
        PrefillEntry(0, C, 2 * C),
        PrefillEntry(1, 5 * C, 6 * C),
        PrefillEntry(2, 0, C),
    ],
    "partial_last": [
        PrefillEntry(0, 0, C),
        PrefillEntry(1, 0, C),
        PrefillEntry(2, 2 * C, 3 * C),
        PrefillEntry(3, 0, 300),
    ],
    # A partial-ending run may precede another slot's full run: the order across slots is free.
    "partial_before_full": [
        PrefillEntry(0, 0, C),
        PrefillEntry(0, C, C + 700),
        PrefillEntry(1, 4 * C, 5 * C),
        PrefillEntry(2, 0, 300),
    ],
    "partial_first": [PrefillEntry(2, 0, 300), PrefillEntry(0, 0, C), PrefillEntry(1, 0, C), EMPTY],
    "empty_tail": [PrefillEntry(0, 2 * C, 3 * C), PrefillEntry(1, 0, 600), EMPTY, EMPTY],
    "mid_slab": [
        PrefillEntry(0, 2 * C + 96, 3 * C + 96),
        PrefillEntry(0, 3 * C + 96, 3 * C + 900),
        PrefillEntry(1, C + 512, 2 * C + 512),
        EMPTY,
    ],
    "empty_any_slot": [PrefillEntry(0, 0, C), PrefillEntry(9, 64, 64), EMPTY, EMPTY],
}


@pytest.mark.parametrize("name", GOOD)
def test_validate_accepts(name):
    entries = GOOD[name]
    validate_forward_entries(entries, chunk_size=C, forward_chunks=len(entries), num_users=4, max_seq_len=8 * C)


BAD = {  # name -> (entries, expected message)
    "wrong_count": ([PrefillEntry(0, 0, C)] * 3, "holds 4 entries"),
    "split_run": (
        [PrefillEntry(0, 0, C), PrefillEntry(1, 0, C), PrefillEntry(0, C, 2 * C), EMPTY],
        "two separate runs",
    ),
    "partial_not_last": ([PrefillEntry(0, 0, 500), PrefillEntry(0, C, 2 * C), EMPTY, EMPTY], "must end its slot"),
    "gap_in_run": ([PrefillEntry(0, 0, C), PrefillEntry(0, 2 * C, 3 * C), EMPTY, EMPTY], "does not continue"),
    "empty_not_tail": ([PrefillEntry(0, 0, C), EMPTY, PrefillEntry(1, 0, C), EMPTY], "after an empty one"),
    "all_empty": ([EMPTY] * 4, "at least one non-empty entry"),
    "negative_start": ([PrefillEntry(0, -32, 32), EMPTY, EMPTY, EMPTY], "negative start"),
    "misaligned_start": ([PrefillEntry(0, 100, 600), EMPTY, EMPTY, EMPTY], "multiple of 32"),
    "more_than_a_chunk": ([PrefillEntry(0, 0, C + 32), EMPTY, EMPTY, EMPTY], "more than one"),
    "end_before_start": ([PrefillEntry(0, 64, 32), EMPTY, EMPTY, EMPTY], "more than one"),
    "cache_overflow": ([PrefillEntry(0, 7 * C + 32, 7 * C + 64), EMPTY, EMPTY, EMPTY], "exceeds the per-user cache"),
    "slot_out_of_range": ([PrefillEntry(4, 0, C), EMPTY, EMPTY, EMPTY], "slot out of range"),
    "shutdown_sentinel": ([PrefillEntry(-1, -1, -1)] * 4, "at least one non-empty entry"),
}


@pytest.mark.parametrize("name", BAD)
def test_validate_rejects(name, expect_error):
    entries, message = BAD[name]
    with expect_error(ValueError, message):
        validate_forward_entries(entries, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)


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
    assert packed_forward_refusals(0, supports_packed_forward=True)
    reasons = packed_forward_refusals(
        4, supports_packed_forward=False, mtp_levels=1, dflash=True, use_trace=True, d2h_acks=True
    )
    assert len(reasons) == 5
    for knob in ("supports_packed_forward", "MTP", "DFLASH", "USE_TRACE", "LAYER_ACK_D2H"):
        assert any(knob in r for r in reasons), knob
