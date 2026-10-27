# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full K/V-head chain sweep: unicast (12x10, 11x10) and multicast (8x8, 8x7, 8x4) chains across chunk sizes, batches,
masks, sinks, padded sequences, head ratios and MLA's separate K and V heads. Grids larger than the device are
skipped."""

import pytest

from tests.ttnn.unit_tests.operations.sdpa.kv_head_chains_test_utils import (
    check_kv_head_chains_gqa,
    check_kv_head_chains_mask,
    check_kv_head_chains_match_q_head_chains,
    check_kv_head_chains_mla_distinct_kv_heads,
    check_kv_head_chains_padded_sq,
    check_kv_head_chains_ratios,
    check_kv_head_chains_sink,
)


@pytest.mark.parametrize("grid", [(8, 8), (8, 7), (8, 4), (12, 10), (11, 10)])
@pytest.mark.parametrize("q_chunk, k_chunk", [(128, 256), (256, 256), (256, 512)])
@pytest.mark.timeout(600)
def test_kv_head_chains_match_q_head_chains(device, grid, q_chunk, k_chunk):
    check_kv_head_chains_match_q_head_chains(device, grid, q_chunk, k_chunk)


@pytest.mark.parametrize("grid", [(8, 8), (8, 7), (12, 10)])
@pytest.mark.parametrize("q_chunk, k_chunk", [(128, 256), (256, 512), (192, 512), (160, 256)])
@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.timeout(600)
def test_kv_head_chains_gqa(device, b, grid, q_chunk, k_chunk):
    check_kv_head_chains_gqa(device, b, grid, q_chunk, k_chunk)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(600)
def test_kv_head_chains_mask(device, grid):
    check_kv_head_chains_mask(device, grid)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(600)
def test_kv_head_chains_sink(device, grid):
    check_kv_head_chains_sink(device, grid)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.parametrize("q_chunk", [128, 256])
@pytest.mark.timeout(600)
def test_kv_head_chains_padded_sq(device, grid, q_chunk):
    check_kv_head_chains_padded_sq(device, grid, q_chunk)


@pytest.mark.parametrize("nh, nkv, s", [(32, 32, 512), (64, 8, 1024), (28, 4, 2048)])
@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(900)
def test_kv_head_chains_ratios(device, nh, nkv, s, grid):
    check_kv_head_chains_ratios(device, nh, nkv, s, grid)


# (24, 4, 3): 6 and 8 Q heads per K / V head do not nest, so the chain key falls back to one Q head
@pytest.mark.parametrize("nh, nkv, nvh", [(32, 1, 32), (32, 2, 8), (32, 8, 2), (24, 4, 3)])
@pytest.mark.timeout(600)
def test_kv_head_chains_mla_distinct_kv_heads(device, nh, nkv, nvh):
    check_kv_head_chains_mla_distinct_kv_heads(device, nh, nkv, nvh)
