# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only FlashMLA-decode latency sweep (single-chip, memory-bound) for TEN-4716.

Mirrors analysis/decode_latency_sweep.py but calls flash_multi_latent_attention_decode. MLA decode
streams the latent KV cache (d_qk = kv_lora + rope) once per step and reuses K as V, so the binding
roof is the K DRAM read. Host-timed over many iters with one device sync. Compare vs predict_decode
(is_mla). Env: MLAD_B, MLAD_NH, MLAD_KVLORA, MLAD_DROPE, MLAD_SEQS (csv), MLAD_ITERS.
"""
from __future__ import annotations
import os
import time
import pytest


def test_mla_decode_latency(device):
    import torch
    import ttnn

    b = int(os.environ.get("MLAD_B", "1"))
    nh = int(os.environ.get("MLAD_NH", "16"))
    nkv = 1
    kv_lora = int(os.environ.get("MLAD_KVLORA", "512"))
    d_rope = int(os.environ.get("MLAD_DROPE", "64"))
    d_qk = kv_lora + d_rope
    seqs = [int(x) for x in os.environ.get("MLAD_SEQS", "1024,2048,4096,8192,16384").split(",")]
    iters = int(os.environ.get("MLAD_ITERS", "80"))

    def _n(v, n=32):
        return ((v + n - 1) // n) * n

    grid = device.compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)
    scale = d_qk**-0.5
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    print(f"\n[mla_decode_sweep] b={b} nh={nh} nkv={nkv} d_qk={d_qk} d_v={kv_lora} iters={iters}", flush=True)
    print(f"{'cache_len':>10}{'k_MB':>10}{'lat_us':>10}{'GB/s':>10}", flush=True)

    for s in seqs:
        k = torch.randn(b, nkv, s, d_qk)
        v = k[..., :kv_lora].contiguous()  # latent V = K slice
        tt_k = ttnn.from_torch(k, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        tt_v = ttnn.from_torch(v, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        q = torch.randn(1, b, nh, d_qk)
        tt_q = ttnn.from_torch(q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        cur_pos = ttnn.Tensor(torch.tensor([s - 1 for _ in range(b)]), ttnn.int32).to(device)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size, q_chunk_size=_n(nh, 32), k_chunk_size=128, exp_approx_mode=False
        )

        for _ in range(3):
            o = ttnn.transformer.flash_multi_latent_attention_decode(
                tt_q,
                tt_k,
                tt_v,
                head_dim_v=kv_lora,
                cur_pos_tensor=cur_pos,
                scale=scale,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=dram,
            )
            o.deallocate()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            o = ttnn.transformer.flash_multi_latent_attention_decode(
                tt_q,
                tt_k,
                tt_v,
                head_dim_v=kv_lora,
                cur_pos_tensor=cur_pos,
                scale=scale,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=dram,
            )
            o.deallocate()
        ttnn.synchronize_device(device)
        lat_us = (time.perf_counter() - t0) / iters * 1e6
        k_bytes = b * nkv * s * d_qk * 2  # K only (bf16), latent reuse
        gbps = k_bytes / (lat_us * 1e-6) / 1e9
        print(f"{s:>10}{k_bytes/1e6:>10.1f}{lat_us:>10.1f}{gbps:>10.1f}", flush=True)
        tt_q.deallocate()
        tt_k.deallocate()
        tt_v.deallocate()
    print("[mla_decode_sweep] OK", flush=True)
