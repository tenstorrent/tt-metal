# SPDX-License-Identifier: Apache-2.0
"""Standalone SDPA chunk-config probe at the exact DP head-fold shape.
Measures DEVICE KERNEL DURATION (via tracy signposts), not wall clock.

Shape: B6 / HQ32 / HKV16 / Sq4096 / Sk8192 / D64, BFP8, LoFi, fp32_dest_acc=True
(matches the in-model SDPA compute_kernel_config). Each config runs under its
own signpost so tt-perf-report can isolate per-config device-kernel time.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(*a, **k):
        return None


B, HQ, HKV, Sq, Sk, DH = 6, 32, 16, 4096, 8192, 64
N_ITERS = 10


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000, "num_command_queues": 1}], indirect=True)
def test_sdpa_chunk_probe(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 required")

    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,  # matches in-model SDPA (compute_common.hpp branch)
    )

    def mk(H, s, dt):
        return ttnn.from_torch(
            torch.randn(B, H, s, DH, dtype=torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # configs: (name, qchunk, kchunk, q_dt, k_dt, v_dt)
    BF8 = ttnn.bfloat8_b
    BF4 = ttnn.bfloat4_b
    configs = [
        ("q128_k2048_kbf4", 128, 2048, BF8, BF4, BF8),  # #216 committed control
        # Finding 2: K-BF4 frees K-CB space, so BF8-K OOM results do NOT close
        # the BF4-K contour. Larger Q chunks => fewer Q work units, same K merges.
        ("q160_k2048_kbf4", 160, 2048, BF8, BF4, BF8),  # 18.8% fewer Q units
        ("q192_k2048_kbf4", 192, 2048, BF8, BF4, BF8),  # 31.2% fewer Q units
        ("q224_k2048_kbf4", 224, 2048, BF8, BF4, BF8),  # only if q192 fits
        ("q256_k2048_kbf4", 256, 2048, BF8, BF4, BF8),  # upper edge probe
    ]

    for name, qc, kc, qdt, kdt, vdt in configs:
        try:
            q = mk(HQ, Sq, qdt)
            k = mk(HKV, Sk, kdt)
            v = mk(HKV, Sk, vdt)
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=qc,
                k_chunk_size=kc,
                exp_approx_mode=True,
            )
            # warmup (compile) outside signpost
            for _ in range(2):
                o = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=1.0, program_config=pc, compute_kernel_config=ck
                )
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh_device)
            signpost(name)
            for _ in range(N_ITERS):
                o = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=1.0, program_config=pc, compute_kernel_config=ck
                )
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"OK {name}")
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        except Exception as e:
            logger.error(f"FAIL {name}: {str(e)[:120]}")
            try:
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            except Exception:
                pass
    signpost("end")
