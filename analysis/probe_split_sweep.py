# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe only: sdpa_decode device time per call (one call traced, replayed), for the V NoC split A/B.
Run once with SDPA_PROBE_SPLIT_KV_NOC=1 and once with =0; lines start with PROBE_SWEEP."""
import os
import statistics
import time

import pytest
import torch
import ttnn

# name, batch, kv heads, K/V dtype, cache length, attended position, grid (None: full), paged block size
# 32 q heads, head dim 128. The full grid cases use k chunk 128 and HiFi2; the 8x8 ones the Llama decode config.
CASES = [
    ("b8_bfp8_2k", 8, 8, "bfloat8_b", 2048, 2047, None, 0),
    ("b8_bfp8_8k", 8, 8, "bfloat8_b", 8192, 8191, None, 0),
    ("b8_bfp8_32k", 8, 8, "bfloat8_b", 32768, 32767, None, 0),
    ("b8_bfp4_16k", 8, 8, "bfloat4_b", 16384, 16383, None, 0),
    ("b8_bfp4_32k", 8, 8, "bfloat4_b", 32768, 32767, None, 0),
    ("b4_bfp8_8k", 4, 8, "bfloat8_b", 8192, 8191, None, 0),
    ("b1_bfp8_32k", 1, 8, "bfloat8_b", 32768, 32767, None, 0),
    ("g88_b32_bfp8_1k_p800", 32, 8, "bfloat8_b", 1024, 800, (8, 8), 0),
    ("g88_b32_bfp8_2k", 32, 8, "bfloat8_b", 2048, 2047, (8, 8), 0),
    ("g88_b32_bfp8_4k", 32, 8, "bfloat8_b", 4096, 4095, (8, 8), 0),
    ("g88_b16_bfp8_4k", 16, 8, "bfloat8_b", 4096, 4095, (8, 8), 0),
    ("g88_b8_bfp8_8k", 8, 8, "bfloat8_b", 8192, 8191, (8, 8), 0),
    ("g88_b4_bfp8_8k", 4, 8, "bfloat8_b", 8192, 8191, (8, 8), 0),
    ("g88_b2_bfp8_8k", 2, 8, "bfloat8_b", 8192, 8191, (8, 8), 0),
    ("g88_b1_bfp8_1k_p500", 1, 8, "bfloat8_b", 1024, 500, (8, 8), 0),
    ("g88_b1_bfp8_8k", 1, 8, "bfloat8_b", 8192, 8191, (8, 8), 0),
    ("g88_b1_bfp8_32k_p16k", 1, 8, "bfloat8_b", 32768, 16383, (8, 8), 0),
    ("g88_b32_bfp8_1k_p128", 32, 8, "bfloat8_b", 1024, 128, (8, 8), 0),
    ("g88_b32_bfp8_1k_p300", 32, 8, "bfloat8_b", 1024, 300, (8, 8), 0),
    ("g88_b16_bfp8_1k_p128", 16, 8, "bfloat8_b", 1024, 128, (8, 8), 0),
    ("g88_b8_bfp8_1k_p128", 8, 8, "bfloat8_b", 1024, 128, (8, 8), 0),
    ("g88_b8_bfp8_8k_p500", 8, 8, "bfloat8_b", 8192, 500, (8, 8), 0),
    ("b8_bfp8_1k_p128", 8, 8, "bfloat8_b", 1024, 128, None, 0),
    ("b8_bfp8_8k_p500", 8, 8, "bfloat8_b", 8192, 500, None, 0),
    ("b16_bfp8_2k_p300", 16, 8, "bfloat8_b", 2048, 300, None, 0),
    ("b32_bfp8_2k_p300", 32, 8, "bfloat8_b", 2048, 300, None, 0),
    ("b8_bfp4_32k_p2k", 8, 8, "bfloat4_b", 32768, 2047, None, 0),
    ("pg88_b32_bfp8_1k_p128", 32, 8, "bfloat8_b", 1024, 128, (8, 8), 32),
    ("pg88_b32_bfp8_1k_p300", 32, 8, "bfloat8_b", 1024, 300, (8, 8), 32),
    ("pg88_b32_bfp8_1k_p500", 32, 8, "bfloat8_b", 1024, 500, (8, 8), 32),
    ("pg88_b32_bfp8_1k_p800", 32, 8, "bfloat8_b", 1024, 800, (8, 8), 32),
    ("pg88_b1_bfp8_1k_p500", 1, 8, "bfloat8_b", 1024, 500, (8, 8), 32),
    ("pg88_b1_bfp8_32k_p2k", 1, 8, "bfloat8_b", 32768, 2047, (8, 8), 32),
    ("pg88_b1_bfp8_32k_p8k", 1, 8, "bfloat8_b", 32768, 8191, (8, 8), 32),
    ("pg88_b1_bfp8_32k_p16k", 1, 8, "bfloat8_b", 32768, 16383, (8, 8), 32),
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
def test_probe_split_sweep(device):
    split = os.environ.get("SDPA_PROBE_SPLIT_KV_NOC", "1")
    only = os.environ.get("PROBE_CASES")
    full = device.compute_with_storage_grid_size()
    dram = ttnn.DRAM_MEMORY_CONFIG
    nh, d = 32, 128
    for name, b, nkv, dtype, s, pos, grid, block in CASES:
        if only and name not in only.split(","):
            continue
        g = grid or (full.x, full.y)
        shape = (b * s // block, nkv, block, d) if block else (b, nkv, s, d)
        kv = []
        for seed in (1, 2):
            x = ttnn.rand(
                shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, low=-1.0, high=1.0, seed=seed
            )
            kv.append(ttnn.typecast(x, getattr(ttnn, dtype), memory_config=dram))
            x.deallocate()
        q = ttnn.rand(
            (1, b, nh, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, low=-1.0, high=1.0, seed=3
        )
        cur_pos = ttnn.Tensor(torch.tensor([pos] * b), ttnn.int32).to(device)
        page_table = None
        if block:
            torch.manual_seed(0)
            blocks = torch.randperm(b * s // block).reshape(b, s // block).to(torch.int32)
            page_table = ttnn.Tensor(blocks, ttnn.int32).to(device)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=g,
            q_chunk_size=0 if grid else 32,
            k_chunk_size=0 if grid else 128,
            exp_approx_mode=False,
        )
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=bool(grid),
            fp32_dest_acc_en=bool(grid),
            packer_l1_acc=bool(grid),
        )

        def call():
            if block:
                return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    kv[0],
                    kv[1],
                    page_table_tensor=page_table,
                    cur_pos_tensor=cur_pos,
                    scale=d**-0.5,
                    program_config=pc,
                    compute_kernel_config=ck,
                    memory_config=dram,
                )
            return ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                kv[0],
                kv[1],
                cur_pos_tensor=cur_pos,
                scale=d**-0.5,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=dram,
            )

        call().deallocate()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = call()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        n = 400 if pos * b <= 65536 else 100
        per = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(device, tid)
        print(
            f"PROBE_SWEEP split={split} arch={device.arch()} grid={g} name={name} us={statistics.median(per):.2f} "
            f"min={min(per):.2f} max={max(per):.2f}",
            flush=True,
        )
        out.deallocate()
        for t in kv + [q, cur_pos] + ([page_table] if block else []):
            t.deallocate()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
def test_probe_split_interleave(device):
    """The decode op alternated with a small matmul in one trace, against the matmul alone, to see whether a dynamic
    NoC mode program costs its neighbours anything."""
    split = os.environ.get("SDPA_PROBE_SPLIT_KV_NOC", "1")
    dram = ttnn.DRAM_MEMORY_CONFIG
    nh, d = 32, 128
    a = ttnn.rand((1, 1, 32, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=5)
    w = ttnn.typecast(
        ttnn.rand((1, 1, 4096, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=6),
        ttnn.bfloat8_b,
    )

    def timed(fn, n=200):
        fn().deallocate()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        per = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(device, tid)
        out.deallocate()
        return statistics.median(per)

    def mm():
        return ttnn.matmul(a, w, memory_config=dram)

    t_mm = timed(mm)
    for name, b, s, pos in [("pg88_b32_1k_p500", 32, 1024, 500), ("pg88_b1_1k_p500", 1, 1024, 500)]:
        shape = (b * s // 32, 8, 32, d)
        kv = [ttnn.typecast(ttnn.rand(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=i), ttnn.bfloat8_b) for i in (1, 2)]
        q = ttnn.rand((1, b, nh, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=3)
        cur_pos = ttnn.Tensor(torch.tensor([pos] * b), ttnn.int32).to(device)
        torch.manual_seed(0)
        page_table = ttnn.Tensor(torch.randperm(b * s // 32).reshape(b, s // 32).to(torch.int32), ttnn.int32).to(device)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)

        def sdpa():
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, kv[0], kv[1], page_table_tensor=page_table, cur_pos_tensor=cur_pos, scale=d**-0.5,
                program_config=pc, compute_kernel_config=ck, memory_config=dram)

        def both():
            o = sdpa()
            m = mm()
            o.deallocate()
            return m

        t_sdpa = timed(sdpa)
        t_both = timed(both)
        print(f"PROBE_INTERLEAVE split={split} name={name} sdpa={t_sdpa:.2f} mm={t_mm:.2f} both={t_both:.2f} "
              f"extra={t_both - t_sdpa - t_mm:.2f}", flush=True)
        for t in kv + [q, cur_pos, page_table]:
            t.deallocate()
