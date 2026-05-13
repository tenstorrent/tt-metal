# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Kernel-only benchmark: measures just the fused GDN kernel execution time.

Runs the kernel in a tight loop with sync between iterations.
Uses B=32 (production batch size) to match real workload.
"""

import math
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("B", [32])
@pytest.mark.parametrize("num_cores", [12, 24, 48, 96])
def test_kernel_bench(mesh_device, reset_seeds, ensure_gc, B, num_cores):
    """Benchmark kernel execution time in isolation."""
    device = mesh_device

    # Architecture constants (Qwen3.5-27B TP=4)
    Nk_TP = 4
    Nv_TP = 12
    Dk = 128
    Dv = 128
    repeat_factor = Nv_TP // Nk_TP
    key_dim_tp = Nk_TP * Dk
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv
    num_pairs = B * Nv_TP
    scale = Dk**-0.5

    logger.info(f"Kernel bench: B={B}, num_pairs={num_pairs}, num_cores={num_cores}")

    torch.manual_seed(42)

    def to_tt(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    # Create input tensors (production shapes)
    conv_out_tt = to_tt(torch.randn(1, B, qkv_dim_tp) * 0.1)
    a_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
    b_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
    neg_exp_A_tt = to_tt(-torch.exp(torch.randn(1, 1, Nv_TP) * 0.5))
    dt_bias_tt = to_tt(torch.randn(1, 1, Nv_TP) * 0.1)
    norm_w_tt = to_tt(torch.ones(1, 1, Dv) + torch.randn(1, 1, Dv) * 0.01)
    scale_tt = to_tt(torch.full((1, 1, 1), scale))
    rms_scale_tt = to_tt(torch.full((1, 1, 1), math.sqrt(Dv)))
    rms_eps_tt = to_tt(torch.full((1, 1, 1), Dv * 1e-6))
    state_tt = to_tt(torch.randn(num_pairs, Dk, Dv) * 0.01)
    output_tt = to_tt(torch.zeros(num_pairs, 1, Dv))

    # Warmup (compile)
    gdn_full_fused_inplace(
        conv_out_tt,
        a_tt,
        b_tt,
        neg_exp_A_tt,
        dt_bias_tt,
        norm_w_tt,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        state_tt,
        output_tt,
        num_pairs=num_pairs,
        num_cores=num_cores,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    ttnn.synchronize_device(device)

    # Benchmark
    num_iters = 20
    times = []

    for i in range(num_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()

        gdn_full_fused_inplace(
            conv_out_tt,
            a_tt,
            b_tt,
            neg_exp_A_tt,
            dt_bias_tt,
            norm_w_tt,
            scale_tt,
            rms_scale_tt,
            rms_eps_tt,
            state_tt,
            output_tt,
            num_pairs=num_pairs,
            num_cores=num_cores,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )

        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # Stats (skip first 2 as warmup)
    times = times[2:]
    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)

    logger.info(f"Results (num_cores={num_cores}):")
    for i, t in enumerate(times):
        logger.info(f"  iter {i}: {t:.3f} ms")
    logger.info(f"  avg={avg:.3f} ms, min={mn:.3f} ms, max={mx:.3f} ms")
    logger.info(f"  pairs/core = {num_pairs/num_cores:.1f}")

    # Per-pair and per-tile-op estimates
    pairs_per_core = num_pairs / num_cores
    tile_ops_per_pair = 131  # from our count
    total_tile_ops = tile_ops_per_pair * pairs_per_core
    us_per_tile_op = (avg * 1000) / total_tile_ops

    logger.info(f"  tile_ops/core = {total_tile_ops:.0f}")
    logger.info(f"  us/tile_op = {us_per_tile_op:.1f}")


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_kernel_bench_breakdown(mesh_device, reset_seeds, ensure_gc):
    """Profile kernel with reader/compute/writer timing via NOC counters.

    Runs with different num_pairs to isolate per-pair overhead.
    """
    device = mesh_device

    Nk_TP = 4
    Nv_TP = 12
    Dk = 128
    Dv = 128
    repeat_factor = Nv_TP // Nk_TP
    key_dim_tp = Nk_TP * Dk
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv
    scale = Dk**-0.5

    def to_tt(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    torch.manual_seed(42)

    # Test with varying B to see per-pair scaling
    results = []
    for B in [1, 2, 4, 8, 16, 32]:
        num_pairs = B * Nv_TP
        nc = min(96, num_pairs)

        conv_out_tt = to_tt(torch.randn(1, B, qkv_dim_tp) * 0.1)
        a_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
        b_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
        neg_exp_A_tt = to_tt(-torch.exp(torch.randn(1, 1, Nv_TP) * 0.5))
        dt_bias_tt = to_tt(torch.randn(1, 1, Nv_TP) * 0.1)
        norm_w_tt = to_tt(torch.ones(1, 1, Dv) + torch.randn(1, 1, Dv) * 0.01)
        scale_tt = to_tt(torch.full((1, 1, 1), scale))
        rms_scale_tt = to_tt(torch.full((1, 1, 1), math.sqrt(Dv)))
        rms_eps_tt = to_tt(torch.full((1, 1, 1), Dv * 1e-6))
        state_tt = to_tt(torch.randn(num_pairs, Dk, Dv) * 0.01)
        output_tt = to_tt(torch.zeros(num_pairs, 1, Dv))

        # Warmup
        gdn_full_fused_inplace(
            conv_out_tt,
            a_tt,
            b_tt,
            neg_exp_A_tt,
            dt_bias_tt,
            norm_w_tt,
            scale_tt,
            rms_scale_tt,
            rms_eps_tt,
            state_tt,
            output_tt,
            num_pairs=num_pairs,
            num_cores=nc,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )
        ttnn.synchronize_device(device)

        times = []
        for _ in range(10):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            gdn_full_fused_inplace(
                conv_out_tt,
                a_tt,
                b_tt,
                neg_exp_A_tt,
                dt_bias_tt,
                norm_w_tt,
                scale_tt,
                rms_scale_tt,
                rms_eps_tt,
                state_tt,
                output_tt,
                num_pairs=num_pairs,
                num_cores=nc,
                Nv_TP=Nv_TP,
                Nk_TP=Nk_TP,
                repeat_factor=repeat_factor,
                key_dim_tp=key_dim_tp,
            )
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = sum(times[2:]) / len(times[2:])
        per_pair = avg / (num_pairs / nc)  # time per pair on the bottleneck core
        results.append((B, num_pairs, nc, avg, per_pair))

        # Cleanup
        for t in [
            conv_out_tt,
            a_tt,
            b_tt,
            neg_exp_A_tt,
            dt_bias_tt,
            norm_w_tt,
            scale_tt,
            rms_scale_tt,
            rms_eps_tt,
            state_tt,
            output_tt,
        ]:
            ttnn.deallocate(t)

    print("\n" + "=" * 80)
    print("KERNEL SCALING: time vs num_pairs (isolating per-pair cost)")
    print("=" * 80)
    print(f"  {'B':>3}  {'pairs':>5}  {'cores':>5}  {'pairs/core':>10}  {'total_ms':>10}  {'ms/pair':>10}")
    for B, np, nc, avg, pp in results:
        print(f"  {B:>3}  {np:>5}  {nc:>5}  {np/nc:>10.1f}  {avg:>10.3f}  {pp:>10.3f}")
