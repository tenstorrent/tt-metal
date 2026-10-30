# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Chunk decomposition for non-cached chunked prefill (``Generator.prefill_chunk_sizes``). Host only."""

import pytest

from models.tt_transformers.tt.generator import MIN_PREFILL_CHUNK_SIZE, prefill_chunk_sizes


@pytest.mark.parametrize(
    "real_len, max_chunk, padded_len, expected",
    [
        # The tail is covered by a small chunk instead of a second full-size one.
        (16470, 16384, 32768, [16384, 2048]),
        (12000, 16384, 16384, [8192, 4096]),
        (9000, 16384, 16384, [8192, 2048]),
        # Exact multiples and exact powers of two.
        (16384, 16384, 16384, [16384]),
        (4096, 16384, 4096, [4096]),
        (6144, 16384, 8192, [4096, 2048]),
        # Larger than one max chunk: max-size chunks first, then the decomposed tail.
        (40000, 16384, 49152, [16384, 16384, 8192]),
        # A max chunk equal to the minimum reduces to the previous uniform chunking.
        (16470, 2048, 18432, [2048] * 9),
        (5000, 2048, 6144, [2048] * 3),
    ],
)
def test_prefill_chunk_sizes(real_len, max_chunk, padded_len, expected):
    sizes = prefill_chunk_sizes(real_len, max_chunk, padded_len)
    assert sizes == expected
    assert all(size & (size - 1) == 0 for size in sizes), "chunks are powers of two"
    assert all(MIN_PREFILL_CHUNK_SIZE <= size <= max_chunk for size in sizes)
    assert sum(sizes) >= real_len
    assert sum(sizes) <= padded_len
    assert sizes == sorted(sizes, reverse=True), "largest chunk first"


@pytest.mark.parametrize("real_len, padded_len", [(1, 32), (100, 128), (2048, 2048), (1500, 2048)])
def test_short_prefill_is_one_chunk(real_len, padded_len):
    # Prompts that fit in the minimum chunk keep the single-shot (non-chunked) length.
    assert prefill_chunk_sizes(real_len, 16384, padded_len) == [padded_len]


def test_inconsistent_padding_falls_back_to_one_chunk():
    # padded_len smaller than the rounded-up real length: nothing to decompose safely.
    assert prefill_chunk_sizes(5000, 16384, 4096) == [4096]
