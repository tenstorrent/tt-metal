# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Paged-attention ops at large block sizes (512 to 2048 tokens), against the existing reference checks.

A hybrid attention + GDN model can keep its GDN state in the attention paged cache when one block of the cache is as
large as one GDN state, which takes blocks of about 1024 tokens. Geometry is Qwen3.6-27B at TP 4 per device: 1 KV
head, head_dim 256.
"""

import math

import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.sdpa.test_sdpa_chunked import run_test_chunked_sdpa
from tests.ttnn.nightly.unit_tests.operations.transformers.test_paged_update_cache import (
    run_test_paged_fill_cache,
    run_test_paged_update_cache_decode,
)
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import run_test_sdpa_decode_paged_attention

BLOCK_SIZES = [64, 512, 896, 1024, 2048]


def _round_up(x, m):
    return m * math.ceil(x / m)


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("where", ["block_start", "block_last", "next_block"])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_paged_update_cache_big_blocks(block_size, where, cache_dtype, device):
    cache_idx = {"block_start": 0, "block_last": block_size - 1, "next_block": block_size + 5}[where]
    # The helper writes user i at cache_idx + 17 * i for 32 users.
    max_seq_len = max(2 * block_size, _round_up(cache_idx + 17 * 31 + 1, block_size))
    run_test_paged_update_cache_decode(
        cache_idx, block_size, 256, max_seq_len, 32, 1, ttnn.bfloat16, cache_dtype, device
    )


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("partial", [False, True], ids=["whole_blocks", "partial_last_block"])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_paged_fill_cache_big_blocks(block_size, partial, cache_dtype, device):
    user_seq_len = 2 * block_size + (128 if partial else 0)
    # paged_fill_cache copies tiles as-is, so the input dtype must equal the cache dtype.
    run_test_paged_fill_cache(block_size, 256, user_seq_len, 4 * block_size, 4, 1, cache_dtype, cache_dtype, device)


@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("d", [256])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_paged_sdpa_decode_big_blocks(block_size, d, kv_dtype, device, reset_seeds):
    s = _round_up(8192, block_size)
    run_test_sdpa_decode_paged_attention(
        device,
        8,
        8,
        1,
        s,
        d,
        kv_dtype,
        (8, 4),
        ttnn.bfloat16,
        True,
        block_size=block_size,
        sharded_in=True,
        sharded_out=False,
        sliding_window_size=None,
    )


def _prefill_chunk(block_size):
    # The chunk ends on a block boundary, as with prefix caching, and on the 128-token SDPA chunk grid.
    return _round_up(1024, math.lcm(block_size, 128))


def _chunked_seq_len(block_size, chunk):
    # Chunked SDPA reads the page table in 32-byte sticks: the table width must be a multiple of 8 blocks.
    return _round_up(4 * chunk, math.lcm(chunk, 8 * block_size))


@pytest.mark.parametrize("k_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8", "bf16"])
@pytest.mark.parametrize("d", [256])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_chunked_sdpa_big_blocks(block_size, d, k_dtype, device):
    chunk = _prefill_chunk(block_size)
    run_test_chunked_sdpa(
        device,
        1,
        8,
        1,
        _chunked_seq_len(block_size, chunk),
        d,
        128,
        128,
        chunk,
        block_size,
        ttnn.bfloat16,
        k_dtype,
        False,
        flexible=True,
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 256 * 1024}], indirect=True, ids=["trace"])
@pytest.mark.parametrize("block_size", [896, 1024])
def test_chunked_sdpa_big_blocks_traced(block_size, device):
    chunk = _prefill_chunk(block_size)
    run_test_chunked_sdpa(
        device,
        1,
        8,
        1,
        _chunked_seq_len(block_size, chunk),
        256,
        128,
        128,
        chunk,
        block_size,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        False,
        flexible=True,
        trace=True,
    )


def test_chunked_sdpa_rejects_page_table_width_not_multiple_of_8(device, expect_error):
    block_size = 1024
    chunk = _prefill_chunk(block_size)
    with expect_error(RuntimeError, "multiple of 8"):
        run_test_chunked_sdpa(
            device,
            1,
            8,
            1,
            4 * block_size,
            256,
            128,
            128,
            chunk,
            block_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            False,
            flexible=True,
        )
