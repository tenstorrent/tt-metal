# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""K/V retention, sanity subset: small cases of each kind of call that fit Wormhole and Blackhole grids. Every shape,
grid and chunk size runs in tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_kv_retention_sweep.py."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.kv_retention_test_utils import (
    check_kv_retention,
    check_kv_retention_causal_and_sliding,
    check_kv_retention_mask_and_sink,
    check_kv_retention_windowed,
    run_sdpa,
    skip_if_too_big,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


# 8x7: a core's Q chunks span two KV heads; 8x4 at q_chunk 192 (does not divide s): Q chunks cross Q-head boundaries
@pytest.mark.parametrize("grid, q_chunk, concat", [((8, 7), 128, False), ((8, 4), 192, True)])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention(device, grid, q_chunk, concat):
    check_kv_retention(device, 2, 16, 2, 256, 64, grid, q_chunk, concat, torch_reference=not concat)


@pytest.mark.timeout(600)
def test_sdpa_kv_retention_past_chain_batch_limit(device):
    """Above batch 8 retention runs without K/V chains: batch 9 against each row run alone (chains at batch 1)."""
    check_kv_retention(device, 9, 8, 2, 256, 64, (8, 4), 128, concat=False, torch_reference=True)


@pytest.mark.parametrize("kind", ["mask", "sink"])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_mask_and_sink(device, kind):
    check_kv_retention_mask_and_sink(device, kind, 1)


# 8x8 causal is where causal K/V chains would otherwise turn on (the reader at q64, the writer at q128)
@pytest.mark.parametrize(
    "kind, grid, q_chunk",
    [("causal", (8, 8), 64), ("causal", (8, 8), 128), ("sliding", (8, 7), 64), ("sliding_causal", (8, 4), 128)],
)
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_causal_and_sliding(device, kind, grid, q_chunk):
    check_kv_retention_causal_and_sliding(device, kind, grid, q_chunk)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_windowed(device, is_causal):
    check_kv_retention_windowed(device, is_causal, 64)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_l1_kv(device, grid):
    """Batch 1 with K/V in L1: unicast chains are dropped (12x10) and multicast ones kept (8x8) under retention, while
    the same call with K/V in DRAM keeps its chains. Only where K/V come from differs, so the two must be bit-identical;
    running both also checks that the L1 and DRAM programs do not share a program-cache entry."""
    skip_if_too_big(device, grid)
    nh, nkv, s, d = 32, 8, 512, 128
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, n, s, d) for n in (nh, nkv, nkv))
    in_l1 = run_sdpa(device, q, k, v, grid, 128, s, memory_config=ttnn.L1_MEMORY_CONFIG)
    in_dram = run_sdpa(device, q, k, v, grid, 128, s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(in_l1, in_dram), "L1- and DRAM-resident K/V must give the same output"
    g = nh // nkv
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    )
    assert_with_pcc(ref, in_l1.float(), 0.99)
