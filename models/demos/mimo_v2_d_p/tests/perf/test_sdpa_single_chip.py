# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip SDPA ceiling for the MiMo GA per-chip shape (32 Q heads / 2 KV heads, DK 192 / DV 128 at TP=2):
plain ttnn.transformer.scaled_dot_product_attention (non-causal, Q vs a full K) — isolates kernel compute limits
from the ring. Profile with --profile; signposts ``single_q{Sq}_k{Sk}_qc{qc}_kc{kc}``."""

import os

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("nq,nkv,dk,dv", [(32, 2, 192, 192), (32, 2, 128, 128)], ids=["192-192", "128-128"])
@pytest.mark.parametrize("qc,kc", [(256, 512), (128, 512), (256, 256)])
def test_sdpa_single_chip(device, nq, nkv, dk, dv, qc, kc):
    Sq, Sk = 2048, 16384
    grid = device.compute_with_storage_grid_size()
    q = ttnn.from_torch(torch.randn(1, nq, Sq, dk), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    k = ttnn.from_torch(torch.randn(1, nkv, Sk, dk), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    v = ttnn.from_torch(torch.randn(1, nkv, Sk, dv), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y), q_chunk_size=qc, k_chunk_size=kc,
                                exp_approx_mode=False)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("MIMO_SDPA_FIDELITY", "HiFi2")),
                                                 math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    for it in range(3):
        if it:
            signpost(f"single_d{dk}v{dv}_q{Sq}_k{Sk}_qc{qc}_kc{kc}_start")
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=dk**-0.5, program_config=pc, compute_kernel_config=ckc)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"single_d{dk}v{dv}_q{Sq}_k{Sk}_qc{qc}_kc{kc}_end")
        o.deallocate(True)
