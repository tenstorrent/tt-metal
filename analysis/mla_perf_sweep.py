# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only FlashMLA prefill sweep for perf-counter capture (TEN-4716 MLA validation).

Mirrors analysis/sdpa_sweep.py: builds q/kvpe, calls ttnn.transformer.flash_mla_prefill
in a loop with no torch golden, so the tracy multi-pass perf-counter capture can cycle
through the FPU/SFPU/MATH counter groups. Latent form: q [b, nh, S, kv_lora+rope],
k [b, nkv, S, kv_lora+rope], V = K[..., :kv_lora], head_dim_v = kv_lora.
"""
from __future__ import annotations
import os
import pytest


def _n(v, n=32):
    return ((v + n - 1) // n) * n


@pytest.mark.parametrize("seq_len", [int(x) for x in os.environ.get("MLA_SEQ", "1024,2048,4096,8192").split(",")])
def test_mla_sweep(device, seq_len):
    import torch
    import ttnn

    nh = int(os.environ.get("MLA_NH", "16"))
    nkv = int(os.environ.get("MLA_NKV", "1"))
    kv_lora = int(os.environ.get("MLA_KVLORA", "512"))
    d_rope = int(os.environ.get("MLA_DROPE", "64"))
    d_qk = kv_lora + d_rope
    b = 1
    padded_num_heads = _n(nh, 32)  # q_chunk_size for MLA prefill

    q = torch.randn(b, nh, seq_len, d_qk)
    k = torch.randn(b, nkv, seq_len, d_qk)
    tt_q = ttnn.from_torch(
        q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_k = ttnn.from_torch(
        k, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=padded_num_heads,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    scale = d_qk**-0.5
    print(f"\n[mla_sweep] S={seq_len} nh={nh} nkv={nkv} d_qk={d_qk} d_v={kv_lora}", flush=True)
    for _ in range(int(os.environ.get("MLA_ITERS", "2"))):
        out = ttnn.transformer.flash_mla_prefill(
            tt_q,
            tt_k,
            head_dim_v=kv_lora,
            scale=scale,
            program_config=pc,
            compute_kernel_config=ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            attn_mask=None,
            is_causal=True,
        )
        ttnn.synchronize_device(device)
        out.deallocate()
    tt_q.deallocate()
    tt_k.deallocate()
    print(f"[mla_sweep] OK S={seq_len}", flush=True)
