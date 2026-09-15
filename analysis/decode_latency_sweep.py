# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only SDPA DECODE latency sweep (TEN-4716 P1 validation).

Decode is MEMORY-bound: one query token streams the whole KV cache each step. This measures the
op's device latency vs KV-cache length (and vs num_kv_heads) so we can check the memory-bound
signature (latency ~ linear in KV bytes) and back out the effective DRAM bandwidth to compare
against the roofline's predict_decode. Host-timed over many iterations with a single device sync
so device time dominates. No torch golden.

Env: DEC_B, DEC_NH, DEC_NKV, DEC_D, DEC_SEQS (csv cache lengths), DEC_ITERS.
"""
from __future__ import annotations
import os
import time
import pytest


def test_decode_latency(device):
    import torch
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    b = int(os.environ.get("DEC_B", "8"))
    nh = int(os.environ.get("DEC_NH", "32"))
    nkv = int(os.environ.get("DEC_NKV", "8"))
    d = int(os.environ.get("DEC_D", "128"))
    seqs = [int(x) for x in os.environ.get("DEC_SEQS", "1024,2048,4096,8192,16384").split(",")]
    iters = int(os.environ.get("DEC_ITERS", "80"))

    def _n(v, n=32):
        return ((v + n - 1) // n) * n

    padded_num_heads = _n(nh, 32)
    grid = device.compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)
    scale = d**-0.5
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    print(f"\n[decode_sweep] b={b} nh={nh} nkv={nkv} d={d} grid={grid_size} iters={iters}", flush=True)
    print(f"{'cache_len':>10}{'kv_MB':>10}{'lat_us':>10}{'GB/s':>10}", flush=True)

    for s in seqs:
        k_chunk = 128 if s >= 512 else 32
        K = fa_rand(b, nkv, s, d)
        V = fa_rand(b, nkv, s, d)
        tt_K = ttnn.as_tensor(K, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        tt_V = ttnn.as_tensor(V, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        Q = fa_rand(1, b, nh, d)
        tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        start_indices = [s - 1 for _ in range(b)]  # attend the full cache
        cur_pos = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

        # warmup (compile) then timed loop
        for _ in range(3):
            o = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
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
            o = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos_tensor=cur_pos,
                scale=scale,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=dram,
            )
            o.deallocate()
        ttnn.synchronize_device(device)
        lat_us = (time.perf_counter() - t0) / iters * 1e6
        kv_bytes = b * nkv * s * d * 2 * 2  # K+V, bf16 (2B)
        gbps = kv_bytes / (lat_us * 1e-6) / 1e9
        print(f"{s:>10}{kv_bytes/1e6:>10.1f}{lat_us:>10.1f}{gbps:>10.1f}", flush=True)
        tt_Q.deallocate()
        tt_K.deallocate()
        tt_V.deallocate()
    print("[decode_sweep] OK", flush=True)
