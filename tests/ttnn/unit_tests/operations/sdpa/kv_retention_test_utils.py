# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared runner and checks for the K/V retention tests (unit subset and nightly sweep).

With a single K chunk, SDPA keeps K/V in a core's CBs across its consecutive Q chunks of one KV head (non-causal,
causal, sliding-window and windowed calls) and, where chains are built, takes it from the previous core only once. Which cores read, receive or keep K/V depends on the batch
size, so every row of a batched call must be bit-identical to the same row run alone, and the output must match torch.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_wormhole_b0


def skip_if_wormhole_lofi_hang(q_chunk):
    # LoFi with fp16 dest hangs Wormhole SDPA unless q_chunk is 1, 2 or 4 tiles (also on main):
    # https://github.com/tenstorrent/tt-metal/issues/61772
    if is_wormhole_b0() and q_chunk // 32 not in (1, 2, 4):
        pytest.skip(f"q_chunk {q_chunk} with LoFi + fp16 dest hangs on Wormhole, see #61772")


def run_sdpa(device, q, k, v, grid, q_chunk, k_chunk, concat=False, memory_config=None, is_causal=False, **kwargs):
    skip_if_wormhole_lofi_hang(q_chunk)
    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
        for t in (q, k, v)
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=q_chunk, k_chunk_size=k_chunk
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=is_causal,
        program_config=pc,
        compute_kernel_config=ck,
        output_concat_heads=concat,
        memory_config=memory_config,
        **kwargs,
    )
    return ttnn.to_torch(out)


def skip_if_too_big(device, grid):
    device_grid = device.compute_with_storage_grid_size()
    if grid[0] > device_grid.x or grid[1] > device_grid.y:
        pytest.skip(f"grid {grid} exceeds the device grid {device_grid.x}x{device_grid.y}")


def check_kv_retention(device, b, nh, nkv, s, d, grid, q_chunk, concat, torch_reference=False):
    skip_if_too_big(device, grid)
    if q_chunk > s:
        pytest.skip("q chunk larger than the sequence")
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    got = run_sdpa(device, q, k, v, grid, q_chunk, s, concat)  # k_chunk = s: one K chunk, so K/V is retained
    if s % q_chunk == 0:
        # Q-head crossing is off at every batch size here, so only where K/V come from differs between the calls
        for i in range(b):
            row = run_sdpa(device, q[i : i + 1], k[i : i + 1], v[i : i + 1], grid, q_chunk, s, concat)
            assert torch.equal(row, got[i : i + 1]), f"batch row {i} differs from the same row run alone"
    # The batch-row check above only covers q chunks that divide S. Where it does not run, fall back to the torch
    # reference whatever the caller asked for, so no case is left asserting nothing -- those are exactly the
    # non-dividing ones, where Q-head crossing is live (gqa_crossing_factor returns 1 when q_chunk divides S).
    if torch_reference or s % q_chunk != 0:
        g = nh // nkv
        torch_out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1), is_causal=False
        )
        if concat:
            # [b, nh, s, d] -> the op's concatenated layout [b, 1, s, nh * d]
            torch_out = torch_out.permute(0, 2, 1, 3).reshape(b, 1, s, nh * d)
        assert_with_pcc(torch_out, got[:, :, :s].float(), 0.99)


def check_kv_retention_mask_and_sink(device, kind, b):
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


def check_kv_retention_causal_and_sliding(device, kind, grid, q_chunk):
    """With one K chunk every causal / sliding-window Q chunk reads the whole chunk, so retention stays on. Each batched
    row must match the same row run alone (a different retention layout), and causal must match torch. From 64 cores
    (8x8) causal K/V chains would turn on too (Blackhole; the reader at q64, the writer at q128), and they hang combined
    with retention, so retention turns them off."""
    skip_if_too_big(device, grid)
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


def check_kv_retention_windowed(device, is_causal, q_chunk):
    """Windowed (block-diagonal) attention with one K chunk retains K/V across a core's Q chunks of a head."""
    grid = (8, 4)
    skip_if_too_big(device, grid)
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
