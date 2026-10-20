# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""K/V retention: with a single K chunk, SDPA keeps K/V in a core's CBs across its consecutive Q chunks of one KV head
(non-causal, causal, sliding-window and windowed calls) and, where chains are built, takes it from the previous core
only once. The non-causal cases here fit Wormhole and Blackhole grids; the full non-causal sweep (including 12x10)
is tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_kv_retention_sweep.py."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.kv_retention_test_utils import check_kv_retention, run_sdpa
from tests.ttnn.utils_for_testing import assert_with_pcc


# (b, nh, nkv, s, d): GQA 32/8 at batch 4 and a group of 8 at a smaller head dim
@pytest.mark.parametrize("b, nh, nkv, s, d", [(4, 32, 8, 512, 128), (2, 16, 2, 256, 64)])
# 8x7: a core's Q chunks span two KV heads; 8x4: whole KV heads per core
@pytest.mark.parametrize("grid", [(8, 7), (8, 4)])
# 192 does not divide s: at batch 4 on 8x4 the Q chunks also cross Q-head boundaries within a GQA group
@pytest.mark.parametrize("q_chunk", [256, 128, 192])
@pytest.mark.parametrize("concat", [False, True], ids=["head_major", "concat"])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention(device, b, nh, nkv, s, d, grid, q_chunk, concat):
    check_kv_retention(device, b, nh, nkv, s, d, grid, q_chunk, concat, torch_reference=not concat)


@pytest.mark.timeout(600)
def test_sdpa_kv_retention_past_chain_batch_limit(device):
    """Above batch 8 retention runs without K/V chains: batch 9 against each row run alone (chains at batch 1)."""
    check_kv_retention(device, 9, 8, 2, 256, 64, (8, 4), 128, concat=False, torch_reference=True)


@pytest.mark.parametrize("kind", ["mask", "sink"])
@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_mask_and_sink(device, kind, b):
    """A mask or a sink does not change which K/V a core needs, so retention stays on for them."""
    grid = (8, 4)
    nh, nkv, s, d = 32, 8, 512, 128
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=128, k_chunk_size=s)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    kr, vr = k.repeat_interleave(nh // nkv, dim=1), v.repeat_interleave(nh // nkv, dim=1)
    if kind == "mask":
        mask = torch.zeros(b, 1, s, s)
        mask[..., 3 * s // 4 :] = -1e9  # serving pad mask: the last quarter of the keys is padding
        tmask = ttnn.from_torch(mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, attn_mask=tmask, program_config=pc, compute_kernel_config=ck
        )
        ref = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
    else:
        sink = torch.rand(1, nh, 1, 1) * 4.0
        tsink = ttnn.from_torch(sink, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, attention_sink=tsink, program_config=pc, compute_kernel_config=ck
        )
        # the sink joins the softmax denominator, scaled like the scores (test_sdpa_prefill's reference)
        scores = torch.cat([(q @ kr.transpose(-1, -2)) / d**0.5, sink.expand(b, nh, s, 1) / d**0.5], dim=-1)
        ref = torch.softmax(scores, dim=-1)[..., :s] @ vr
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)


def _skip_if_too_big(device, grid):
    g = device.compute_with_storage_grid_size()
    if grid[0] > g.x or grid[1] > g.y:
        pytest.skip(f"grid {grid} exceeds the device grid {g.x}x{g.y}")


@pytest.mark.parametrize("kind", ["causal", "sliding", "sliding_causal"])
# 8x8: from 64 cores causal K/V chains would turn on too (Blackhole; the reader at q64, the writer at q128), and they
# hang combined with retention, so retention turns them off
@pytest.mark.parametrize("grid", [(8, 7), (8, 4), (8, 8)])
@pytest.mark.parametrize("q_chunk", [64, 128])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_causal_and_sliding(device, kind, grid, q_chunk):
    """With one K chunk every causal / sliding-window Q chunk reads the whole chunk, so retention stays on. Each batched
    row must match the same row run alone (a different retention layout), and causal must match torch."""
    _skip_if_too_big(device, grid)
    b, nh, nkv, s, d = 4, 32, 8, 256, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    kwargs = dict(is_causal=kind != "sliding")
    if kind.startswith("sliding"):
        kwargs["sliding_window_size"] = 128
    got = run_sdpa(device, q, k, v, grid, q_chunk, s, **kwargs)
    for i in range(b):
        row = run_sdpa(device, q[i : i + 1], k[i : i + 1], v[i : i + 1], grid, q_chunk, s, **kwargs)
        assert torch.equal(row, got[i : i + 1]), f"batch row {i} differs from the same row run alone"
    if kind == "causal":
        g = nh // nkv
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1), is_causal=True
        )
        assert_with_pcc(ref, got.float(), 0.99)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("q_chunk", [64, 128])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_windowed(device, is_causal, q_chunk):
    """Windowed (block-diagonal) attention with one K chunk retains K/V across a core's Q chunks of a head."""
    grid = (8, 4)
    _skip_if_too_big(device, grid)
    # 32 heads, not 8: on this 32-core grid 8 heads give at most one Q chunk per core (32 chunks at q_chunk=64,
    # 16 at 128), so no core ever sees two consecutive chunks of a head and keep_kv stays false -- the retention
    # this test is named for never runs. 32 heads give 4 or 2 chunks per core, exercising it and its final release.
    nh, nkv, s, d = 32, 32, 256, 64
    cu = [0, 96, 160, 256]
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, n, s, d) for n in (nh, nkv, nkv))
    cu_tt = ttnn.from_torch(
        torch.tensor(cu, dtype=torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    got = run_sdpa(device, q, k, v, grid, q_chunk, s, is_causal=is_causal, cu_window_seqlens=cu_tt)
    mask = torch.full((s, s), float("-inf"))
    for lo, hi in zip(cu[:-1], cu[1:]):
        mask[lo:hi, lo:hi] = 0.0
    if is_causal:
        mask = mask + torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert_with_pcc(ref, got.float(), 0.99)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_l1_kv(device, grid):
    """Batch 1 with K/V in L1: unicast chains are dropped (12x10) and multicast ones kept (8x8) under retention, while
    the same call with K/V in DRAM keeps its chains. Only where K/V come from differs, so the two must be bit-identical;
    running both also checks that the L1 and DRAM programs do not share a program-cache entry."""
    _skip_if_too_big(device, grid)
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
