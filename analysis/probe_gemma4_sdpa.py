# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 style prefill SDPA shapes (causal, d256 sliding on 8x8, d512 global on 8x4, HiFi4 fp32 DEST, as
models/demos/gemma4 configures them), device time per call from traced replays. Lines start with PROBE_G4."""
import statistics
import time

import pytest
import torch
import ttnn

# name, q heads, kv heads, head dim, seq, sliding window, grid, q chunk, k chunk, dtype
CASES = [
    ("slide_d256_s128", 16, 8, 256, 128, 1024, (8, 8), 128, 128, "bfloat16"),
    ("global_d512_s128", 16, 4, 512, 128, None, (8, 4), 128, 128, "bfloat16"),
    ("slide_d256_s1024", 16, 8, 256, 1024, 1024, (8, 8), 256, 128, "bfloat16"),
    ("global_d512_s1024", 16, 4, 512, 1024, None, (8, 4), 128, 128, "bfloat16"),
    ("slide_d256_s4096", 16, 8, 256, 4096, 1024, (8, 8), 256, 128, "bfloat16"),
    ("global_d512_s4096", 16, 4, 512, 4096, None, (8, 4), 128, 128, "bfloat16"),
    ("slide_d256_s128_bfp8", 16, 8, 256, 128, 1024, (8, 8), 128, 128, "bfloat8_b"),
    ("global_d512_s1024_bfp8", 16, 4, 512, 1024, None, (8, 4), 128, 128, "bfloat8_b"),
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
def test_probe_gemma4_sdpa(device):
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    for name, nh, nkv, d, s, win, grid, qc, kc, dt in CASES:
        dtype = getattr(ttnn, dt)
        q = ttnn.typecast(ttnn.rand((1, nh, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=1), dtype)
        k = ttnn.typecast(ttnn.rand((1, nkv, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=2), dtype)
        v = ttnn.typecast(ttnn.rand((1, nkv, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=3), dtype)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
        )

        def call():
            return ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=d**-0.5, sliding_window_size=win, program_config=pc, compute_kernel_config=ck
            )

        call().deallocate()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = call()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        n = 200 if s <= 1024 else 30
        per = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(device, tid)
        print(f"PROBE_G4 name={name} us={statistics.median(per):.2f} min={min(per):.2f} max={max(per):.2f}", flush=True)
        out.deallocate()
        for t in (q, k, v):
            t.deallocate()
