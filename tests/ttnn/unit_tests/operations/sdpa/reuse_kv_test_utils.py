# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared runner for the ``scaled_dot_product_attention(..., reuse_kv=True)`` tests (unit subset and nightly sweep)."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _run(device, q, k, v, grid, q_chunk, k_chunk, concat, pack, reuse):
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
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
        is_causal=False,
        program_config=pc,
        compute_kernel_config=ck,
        output_concat_heads=concat,
        pack_gqa_heads=pack,
        reuse_kv=reuse,
    )
    return ttnn.to_torch(out)


def check_reuse_kv(device, b, nh, nkv, s, d, grid, q_chunk, concat, pack, torch_reference=False):
    """reuse_kv=True must be bit-identical to reuse_kv=False (only where K/V come from changes)."""
    device_grid = device.compute_with_storage_grid_size()
    if grid[0] > device_grid.x or grid[1] > device_grid.y:
        pytest.skip(f"grid {grid} exceeds the device grid {device_grid.x}x{device_grid.y}")
    if q_chunk > s:
        pytest.skip("q chunk larger than the sequence")
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    ref = _run(device, q, k, v, grid, q_chunk, s, concat, pack, reuse=False)
    got = _run(device, q, k, v, grid, q_chunk, s, concat, pack, reuse=True)
    assert torch.equal(got, ref), "reuse_kv must be bit-identical to reading K/V per Q chunk"
    if torch_reference:
        assert not concat and not pack, "the torch reference is for the head-major, unpacked layout"
        g = nh // nkv
        torch_out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1), is_causal=False
        )
        assert_with_pcc(torch_out, got.float(), 0.99)
