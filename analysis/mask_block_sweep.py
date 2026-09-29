# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Non causal SDPA prefill with a block granular Bernoulli mask, with or without the block map (issue 55761).

Env: MB_S (4096), MB_P masked block fraction (0.5), MB_MAP 1 to pass attn_mask_block_map (1), MB_ITERS (3),
MB_NH (16), MB_NKV (16), MB_D (128), MB_QC / MB_KC (128). Run under tracy, read the SDPA op durations.
"""
import os

import torch


def test_mask_block_sweep(device):
    import ttnn

    s = int(os.environ.get("MB_S", "4096"))
    p = float(os.environ.get("MB_P", "0.5"))
    use_map = os.environ.get("MB_MAP", "1") == "1"
    iters = int(os.environ.get("MB_ITERS", "3"))
    nh = int(os.environ.get("MB_NH", "16"))
    nkv = int(os.environ.get("MB_NKV", "16"))
    d = int(os.environ.get("MB_D", "128"))
    qc = int(os.environ.get("MB_QC", "128"))
    kc = int(os.environ.get("MB_KC", "128"))
    torch.manual_seed(7)
    nq, nk = s // qc, s // kc
    masked = torch.bernoulli(torch.full((1, 1, nq, nk), p))
    masked[..., 0] = 0
    mask = masked.repeat_interleave(qc, dim=2).repeat_interleave(kc, dim=3) * -1e9
    block_map = (masked == 0).to(torch.int32)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_map = ttnn.from_torch(block_map, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) if use_map else None
    q = ttnn.from_torch(torch.randn(1, nh, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(torch.randn(1, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(torch.randn(1, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=True
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    print(f"\n[mask_block_sweep] S={s} p={p} map={use_map} active={int(block_map.sum())}/{nq * nk}", flush=True)
    for _ in range(iters):
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=tt_mask, program_config=pc, compute_kernel_config=ck, attn_mask_block_map=tt_map
        )
        ttnn.synchronize_device(device)
        out.deallocate()
    print("[mask_block_sweep] OK", flush=True)
