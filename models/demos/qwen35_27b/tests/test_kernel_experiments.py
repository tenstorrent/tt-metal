# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Kernel optimization experiments: isolate the impact of each approach.

Experiment 0: BASELINE — DRAM state, Dv=128 (current production)
Experiment 1: L1_STATE — state in L1 interleaved, Dv=128
Experiment 2: V_BLOCK  — DRAM state, Dv=32 (Vt=1, 4× less state I/O)
Experiment 3: L1+VBLOCK — state in L1 + Dv=32 (compute-only floor)
"""

import math
import os
import time

import pytest
import torch

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace


def _run_kernel_bench(device, B, Dv, state_in_l1, num_cores, label, num_iters=20):
    """Run kernel benchmark with specified configuration."""
    Nk_TP = 4
    Nv_TP = 12
    Dk = 128
    repeat_factor = Nv_TP // Nk_TP
    key_dim_tp = Nk_TP * Dk  # 512
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv  # varies with Dv
    num_pairs = B * Nv_TP
    scale = Dk**-0.5
    Vt = Dv // 32

    torch.manual_seed(42)

    def to_tt(t, l1=False):
        mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )

    conv_out_tt = to_tt(torch.randn(1, B, qkv_dim_tp) * 0.1)
    a_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
    b_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
    neg_exp_A_tt = to_tt(-torch.exp(torch.randn(1, 1, Nv_TP) * 0.5))
    dt_bias_tt = to_tt(torch.randn(1, 1, Nv_TP) * 0.1)
    norm_w_tt = to_tt(torch.ones(1, 1, Dv) + torch.randn(1, 1, Dv) * 0.01)
    scale_tt = to_tt(torch.full((1, 1, 1), scale))
    rms_scale_tt = to_tt(torch.full((1, 1, 1), math.sqrt(Dv)))
    rms_eps_tt = to_tt(torch.full((1, 1, 1), Dv * 1e-6))
    state_tt = to_tt(torch.randn(num_pairs, Dk, Dv) * 0.01, l1=state_in_l1)
    output_tt = to_tt(torch.zeros(num_pairs, 1, Dv))

    # Warmup / compile
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
    times = []
    for _ in range(num_iters):
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

    # Skip first 2
    times = times[2:]
    avg = sum(times) / len(times)
    mn = min(times)

    state_tiles = (Dk // 32) * Vt
    state_bytes = num_pairs * state_tiles * 2048
    state_loc = "L1" if state_in_l1 else "DRAM"

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

    return {
        "label": label,
        "avg_ms": avg,
        "min_ms": mn,
        "Dv": Dv,
        "Vt": Vt,
        "state_tiles": state_tiles,
        "state_MB": state_bytes / 1e6,
        "state_loc": state_loc,
        "num_cores": num_cores,
        "pairs_per_core": num_pairs / num_cores,
    }


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_kernel_experiments(mesh_device, reset_seeds, ensure_gc):
    """Run all 4 experiments and compare."""
    device = mesh_device
    B = 32
    num_pairs = B * 12  # 384

    experiments = []

    # Sweep num_cores for each config
    for nc in [12, 24, 48, 96]:
        # Exp 0: BASELINE (DRAM, Dv=128)
        r = _run_kernel_bench(device, B, Dv=128, state_in_l1=False, num_cores=nc, label=f"BASELINE(DRAM,Dv128,c{nc})")
        experiments.append(r)

    for nc in [12, 24, 48, 96]:
        # Exp 1: L1 STATE (L1, Dv=128)
        r = _run_kernel_bench(device, B, Dv=128, state_in_l1=True, num_cores=nc, label=f"L1_STATE(L1,Dv128,c{nc})")
        experiments.append(r)

    for nc in [12, 24, 48, 96]:
        # Exp 2: V-BLOCK proxy (DRAM, Dv=32)
        r = _run_kernel_bench(device, B, Dv=32, state_in_l1=False, num_cores=nc, label=f"V_BLOCK(DRAM,Dv32,c{nc})")
        experiments.append(r)

    for nc in [12, 24, 48, 96]:
        # Exp 3: L1 + V-BLOCK (L1, Dv=32) — compute floor
        r = _run_kernel_bench(device, B, Dv=32, state_in_l1=True, num_cores=nc, label=f"L1+VBLK(L1,Dv32,c{nc})")
        experiments.append(r)

    # Print results
    print("\n" + "=" * 100)
    print("KERNEL OPTIMIZATION EXPERIMENTS (B=32, 384 pairs, single device)")
    print("=" * 100)
    print(
        f"  {'Label':<30} {'cores':>5} {'p/core':>6} {'Dv':>4} {'state':>5} "
        f"{'st_tiles':>8} {'st_MB':>6} {'avg_ms':>8} {'min_ms':>8}"
    )
    print("-" * 100)

    for r in experiments:
        print(
            f"  {r['label']:<30} {r['num_cores']:>5} {r['pairs_per_core']:>6.0f} "
            f"{r['Dv']:>4} {r['state_loc']:>5} {r['state_tiles']:>8} "
            f"{r['state_MB']:>6.1f} {r['avg_ms']:>8.3f} {r['min_ms']:>8.3f}"
        )

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY (best num_cores for each config)")
    print("=" * 100)

    configs = {}
    for r in experiments:
        key = r["label"].split("(")[0]  # BASELINE, L1_STATE, V_BLOCK, L1+VBLK
        if key not in configs or r["min_ms"] < configs[key]["min_ms"]:
            configs[key] = r

    baseline_ms = configs.get("BASELINE", {}).get("min_ms", 1.0)
    for key in ["BASELINE", "L1_STATE", "V_BLOCK", "L1+VBLK"]:
        if key in configs:
            r = configs[key]
            speedup = baseline_ms / r["min_ms"]
            print(
                f"  {key:<12}: {r['min_ms']:.3f} ms  ({speedup:.2f}x vs baseline)  "
                f"[cores={r['num_cores']}, Dv={r['Dv']}, state={r['state_loc']}, "
                f"state_tiles={r['state_tiles']}, state_MB={r['state_MB']:.1f}]"
            )


def _run_kernel_bench_mesh(mesh_device, B, Dv, state_in_l1, num_cores, label, num_iters=20):
    """Run kernel benchmark on multi-device mesh."""
    Nk_TP = 4
    Nv_TP = 12
    Dk = 128
    repeat_factor = Nv_TP // Nk_TP
    key_dim_tp = Nk_TP * Dk
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv
    num_pairs = B * Nv_TP
    scale = Dk**-0.5

    torch.manual_seed(42)

    def to_tt(t, l1=False):
        mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=mc,
        )

    conv_out_tt = to_tt(torch.randn(1, B, qkv_dim_tp) * 0.1)
    a_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
    b_tt = to_tt(torch.randn(1, B, Nv_TP) * 0.5)
    neg_exp_A_tt = to_tt(-torch.exp(torch.randn(1, 1, Nv_TP) * 0.5))
    dt_bias_tt = to_tt(torch.randn(1, 1, Nv_TP) * 0.1)
    norm_w_tt = to_tt(torch.ones(1, 1, Dv) + torch.randn(1, 1, Dv) * 0.01)
    scale_tt = to_tt(torch.full((1, 1, 1), scale))
    rms_scale_tt = to_tt(torch.full((1, 1, 1), math.sqrt(Dv)))
    rms_eps_tt = to_tt(torch.full((1, 1, 1), Dv * 1e-6))
    state_tt = to_tt(torch.randn(num_pairs, Dk, Dv) * 0.01, l1=state_in_l1)
    output_tt = to_tt(torch.zeros(num_pairs, 1, Dv))

    # Warmup / compile
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
    ttnn.synchronize_device(mesh_device)

    times = []
    for _ in range(num_iters):
        ttnn.synchronize_device(mesh_device)
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
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = times[2:]
    avg = sum(times) / len(times)
    mn = min(times)
    state_tiles = (Dk // 32) * (Dv // 32)
    state_bytes = num_pairs * state_tiles * 2048

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

    return {
        "label": label,
        "avg_ms": avg,
        "min_ms": mn,
        "Dv": Dv,
        "Vt": Dv // 32,
        "state_tiles": state_tiles,
        "state_MB": state_bytes / 1e6,
        "state_loc": "L1" if state_in_l1 else "DRAM",
        "num_cores": num_cores,
        "pairs_per_core": num_pairs / num_cores,
    }


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
def test_kernel_experiments_mesh(mesh_device, reset_seeds, ensure_gc):
    """Run experiments on multi-device mesh (production config)."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < 4:
        pytest.skip("Need TP>=4 for mesh experiment")

    B = 32
    experiments = []

    for nc in [12, 24, 48, 96]:
        r = _run_kernel_bench_mesh(
            mesh_device, B, Dv=128, state_in_l1=False, num_cores=nc, label=f"BASELINE(DRAM,Dv128,c{nc})"
        )
        experiments.append(r)

    for nc in [12, 24, 48, 96]:
        r = _run_kernel_bench_mesh(
            mesh_device, B, Dv=128, state_in_l1=True, num_cores=nc, label=f"L1_STATE(L1,Dv128,c{nc})"
        )
        experiments.append(r)

    for nc in [12, 24, 48, 96]:
        r = _run_kernel_bench_mesh(
            mesh_device, B, Dv=32, state_in_l1=False, num_cores=nc, label=f"V_BLOCK(DRAM,Dv32,c{nc})"
        )
        experiments.append(r)

    for nc in [12, 24, 48, 96]:
        r = _run_kernel_bench_mesh(
            mesh_device, B, Dv=32, state_in_l1=True, num_cores=nc, label=f"L1+VBLK(L1,Dv32,c{nc})"
        )
        experiments.append(r)

    print(f"\n{'=' * 100}")
    print(f"KERNEL OPTIMIZATION EXPERIMENTS (B=32, 384 pairs, {num_devices}-device mesh)")
    print("=" * 100)
    print(
        f"  {'Label':<30} {'cores':>5} {'p/core':>6} {'Dv':>4} {'state':>5} "
        f"{'st_tiles':>8} {'st_MB':>6} {'avg_ms':>8} {'min_ms':>8}"
    )
    print("-" * 100)

    for r in experiments:
        print(
            f"  {r['label']:<30} {r['num_cores']:>5} {r['pairs_per_core']:>6.0f} "
            f"{r['Dv']:>4} {r['state_loc']:>5} {r['state_tiles']:>8} "
            f"{r['state_MB']:>6.1f} {r['avg_ms']:>8.3f} {r['min_ms']:>8.3f}"
        )

    print(f"\n{'=' * 100}")
    print("SUMMARY (best num_cores for each config)")
    print("=" * 100)

    configs = {}
    for r in experiments:
        key = r["label"].split("(")[0]
        if key not in configs or r["min_ms"] < configs[key]["min_ms"]:
            configs[key] = r

    baseline_ms = configs.get("BASELINE", {}).get("min_ms", 1.0)
    for key in ["BASELINE", "L1_STATE", "V_BLOCK", "L1+VBLK"]:
        if key in configs:
            r = configs[key]
            speedup = baseline_ms / r["min_ms"]
            print(
                f"  {key:<12}: {r['min_ms']:.3f} ms  ({speedup:.2f}x vs baseline)  "
                f"[cores={r['num_cores']}, Dv={r['Dv']}, state={r['state_loc']}, "
                f"state_tiles={r['state_tiles']}, state_MB={r['state_MB']:.1f}]"
            )
