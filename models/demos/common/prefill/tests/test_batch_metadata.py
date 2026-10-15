# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill metadata wire under both layouts (host-only, no ttnn)."""

import struct

import pytest

from models.demos.common.prefill.batch_metadata import (
    BATCH_METADATA_BYTES,
    INVALID_SLOT,
    MAX_WIRE_CHUNKS,
    RECORD_BYTES,
    SHUTDOWN_WORD,
    WARMUP_WORD,
    ChunkRecord,
    decode,
    encode,
    encode_words,
    is_sentinel,
    metadata_size_bytes,
    sentinel,
    sentinel_words,
)


def test_sizes_match_the_engine_structs(expect_error):
    assert RECORD_BYTES == 12
    assert BATCH_METADATA_BYTES == 8 + 12 * MAX_WIRE_CHUNKS == 200
    assert metadata_size_bytes(1) == 12
    assert metadata_size_bytes(2) == 200
    assert metadata_size_bytes(MAX_WIRE_CHUNKS) == 200
    with expect_error(ValueError, "max_chunks=0 outside"):
        metadata_size_bytes(0)
    with expect_error(ValueError, f"max_chunks={MAX_WIRE_CHUNKS + 1} outside"):
        metadata_size_bytes(MAX_WIRE_CHUNKS + 1)


def test_single_chunk_layout_is_the_bare_record():
    rec = ChunkRecord(3, 1024, 1500)
    assert encode([rec], 1) == struct.pack("<3I", 3, 1024, 1500)
    assert decode([3, 1024, 1500], 1) == [rec]


def test_batched_layout_round_trips_and_fills_the_rest_with_invalid():
    recs = [ChunkRecord(0, 0, 1024), ChunkRecord(1, 1024, 2048), ChunkRecord(1, 2048, 2500)]
    words = encode_words(recs, 8)
    assert words[:2] == [3, 0]
    assert words[2:11] == [0, 0, 1024, 1, 1024, 2048, 1, 2048, 2500]
    assert words[11:] == [INVALID_SLOT, 0, 0] * (MAX_WIRE_CHUNKS - 3)
    assert len(encode(recs, 8)) == BATCH_METADATA_BYTES
    assert decode(words, 8) == recs


def test_decode_accepts_the_signed_view_a_torch_int32_tensor_gives():
    recs = [ChunkRecord(2, 96, 150)]
    signed = [w - (1 << 32) if w >= 1 << 31 else w for w in encode_words(recs, 4)]
    assert signed[5] == -1  # the first unused record's INVALID_SLOT reads as -1 when signed
    assert decode(signed, 4) == recs


def test_decode_rejects_bad_counts_and_short_pushes(expect_error):
    with expect_error(ValueError, "num_chunks=0 outside"):
        decode([0, 0] + [0] * 48, 4)
    with expect_error(ValueError, "num_chunks=5 outside"):
        decode([5, 0] + [0] * 48, 4)
    with expect_error(ValueError, "3 metadata words, need 50"):
        decode([1, 0, 0], 4)
    with expect_error(ValueError, "3 records for a forward of at most 2 chunks"):
        encode([ChunkRecord(0, 0, 1)] * 3, 2)
    with expect_error(ValueError, "0 records for a forward of at most 2 chunks"):
        encode([], 2)


@pytest.mark.parametrize("max_chunks", [1, 2, MAX_WIRE_CHUNKS])
@pytest.mark.parametrize("word", [SHUTDOWN_WORD, WARMUP_WORD])
def test_sentinels_read_the_same_under_both_layouts(max_chunks, word):
    words = sentinel_words(word, max_chunks)
    assert len(sentinel(word, max_chunks)) == metadata_size_bytes(max_chunks)
    assert is_sentinel(words, word)
    assert not is_sentinel(words, WARMUP_WORD if word == SHUTDOWN_WORD else SHUTDOWN_WORD)
    signed = [w - (1 << 32) for w in words]
    assert is_sentinel(signed, word)


def test_a_real_push_is_never_a_sentinel():
    assert not is_sentinel(encode_words([ChunkRecord(0, 0, 8)], 1), SHUTDOWN_WORD)
    assert not is_sentinel(encode_words([ChunkRecord(INVALID_SLOT - 1, 0, 8)], 4), SHUTDOWN_WORD)
