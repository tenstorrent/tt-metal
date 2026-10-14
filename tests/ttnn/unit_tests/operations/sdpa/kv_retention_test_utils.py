# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared runner for the K/V retention tests (unit subset and nightly sweep).

With a single K chunk, non-causal SDPA keeps K/V in a core's CBs across its consecutive Q chunks of one KV head and
takes it from the previous core in its chain only once. Which cores read, receive or keep K/V depends on the batch
size, so every row of a batched call must be bit-identical to the same row run alone, and the output must match torch.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_sdpa(device, q, k, v, grid, q_chunk, k_chunk, concat=False, memory_config=None, is_causal=False, **kwargs):
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


def check_kv_retention(device, b, nh, nkv, s, d, grid, q_chunk, concat, torch_reference=False):
    device_grid = device.compute_with_storage_grid_size()
    if grid[0] > device_grid.x or grid[1] > device_grid.y:
        pytest.skip(f"grid {grid} exceeds the device grid {device_grid.x}x{device_grid.y}")
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
