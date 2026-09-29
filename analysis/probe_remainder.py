# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Causal prefill shapes with a minority q chunk remainder (the #56897 table and neighbours), device time per call
from traced replays; SDPA_PROBE_REMAINDER=0 keeps main's placement on the probe build. Lines start with PROBE_RM."""
import os
import statistics
import time

import pytest
import ttnn

# name, q heads, kv heads, head dim, seq, grid, q chunk, k chunk, K/V dtype, fp32 DEST
CASES = [
    ("nh32_d128_bfp8_s1024", 32, 8, 128, 1024, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh32_d128_bfp8_s2048", 32, 8, 128, 2048, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh32_d128_bfp8_s4096", 32, 8, 128, 4096, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh32_d128_bfp8_s8192", 32, 8, 128, 8192, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh32_d128_bf16_s1024", 32, 8, 128, 1024, (11, 10), 128, 128, "bfloat16", False),
    ("nh32_d128_bf16_s8192", 32, 8, 128, 8192, (11, 10), 128, 128, "bfloat16", False),
    ("nh8_d128_bfp8_s4096", 8, 1, 128, 4096, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh8_d128_bfp8_s16384", 8, 1, 128, 16384, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh16_d128_bfp8_s4096", 16, 4, 128, 4096, (11, 10), 128, 128, "bfloat8_b", False),
    ("nh16_d256_bf16_s4096", 16, 8, 256, 4096, (11, 10), 128, 128, "bfloat16", False),
    ("nh32_d128_bfp8_s1024_q64", 32, 8, 128, 1024, (11, 10), 64, 64, "bfloat8_b", False),
    ("nh32_d128_bfp8_s4096_q256", 32, 8, 128, 4096, (11, 10), 256, 256, "bfloat8_b", False),
    ("nh32_d128_bfp8_s1024_fp32", 32, 8, 128, 1024, (11, 10), 64, 64, "bfloat8_b", True),
    ("nh32_d128_bfp8_s1024_g88", 32, 8, 128, 1024, (8, 8), 128, 128, "bfloat8_b", False),
    ("nh12_d64_bf16_s2048", 12, 12, 64, 2048, (11, 10), 128, 128, "bfloat16", False),
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
def test_probe_remainder(device):
    tag = os.environ.get("PROBE_TAG", "x")
    for name, nh, nkv, d, s, grid, qc, kc, kv, fp32 in CASES:
        ck = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4 if fp32 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        q = ttnn.rand((1, nh, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=1)
        kvs = []
        for seed in (2, 3):
            x = ttnn.rand((1, nkv, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=seed)
            kvs.append(ttnn.typecast(x, getattr(ttnn, kv)))
            x.deallocate()
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
        )

        def call():
            return ttnn.transformer.scaled_dot_product_attention(
                q, kvs[0], kvs[1], is_causal=True, scale=d**-0.5, program_config=pc, compute_kernel_config=ck
            )

        call().deallocate()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = call()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        n = 40 if s <= 2048 else 8
        per = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(device, tid)
        print(f"PROBE_RM tag={tag} name={name} us={statistics.median(per):.2f} min={min(per):.2f}", flush=True)
        out.deallocate()
        for t in [q] + kvs:
            t.deallocate()
