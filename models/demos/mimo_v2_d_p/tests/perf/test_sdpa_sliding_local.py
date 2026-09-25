# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip plain sliding SDPA (window 128 + sink) on the MiMo SWA per-chip shape at TP=2 (32 Q / 4 KV heads,
D 192): Q front-padded by the 128-token halo, K/V = [halo | local]. Compares against the ring sliding path."""

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


@pytest.mark.parametrize("C", [640, 2048])
@pytest.mark.parametrize("qc,kc", [(128, 128), (256, 256), (128, 256)])
def test_sdpa_sliding_local(device, C, qc, kc):
    S = C + 128
    grid = device.compute_with_storage_grid_size()
    q = ttnn.from_torch(torch.randn(1, 32, S, 192), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    k = ttnn.from_torch(torch.randn(1, 4, S, 192), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    v = ttnn.from_torch(torch.randn(1, 4, S, 192), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    sink = ttnn.from_torch(torch.randn(1, 32, 1, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                 fp32_dest_acc_en=False, packer_l1_acc=False)
    for it in range(3):
        if it:
            signpost(f"slide_C{C}_qc{qc}_kc{kc}_start")
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=192**-0.5, sliding_window_size=128,
                                                          attention_sink=sink, program_config=pc, compute_kernel_config=ckc)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"slide_C{C}_qc{qc}_kc{kc}_end")
        o.deallocate(True)
