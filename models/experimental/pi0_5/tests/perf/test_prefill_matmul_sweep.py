# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep matmul program configs at the pi0.5 VLM-prefill + SigLIP shapes.

These matmuls run on the 2D block-shard path
(`MatmulMultiCoreReuseMultiCastProgramConfig`) because m_tiles >= 8
fills the 12×10 grid. The tunable knobs differ from the 1D path:

  - in0_block_w         {2, 4, 8, 16, 32}  must divide K_tiles
  - per_core_M          (m_tiles + grid_y - 1) // grid_y
  - per_core_N          (n_tiles + grid_x - 1) // grid_x
  - grid                {(12,10), (8,8), (12,8)}  — fewer cores = bigger per-core
  - dst_budget          {4, 6, 8}
  - fp32_dest_acc_en    {False, True}
  - math_fidelity       {LoFi, HiFi2, HiFi4}
  - transpose_mcast     {False, True}  — direction of multicast

Each config verified PCC vs torch reference (default ≥ 0.9999).

Shapes covered:
  VLM (M=512):
    M=512, K=2048,  N=2560    QKV fused          (1 per layer × 18 = 18 calls)
    M=512, K=2048,  N=2048    O projection       (18 calls)
    M=512, K=2048,  N=16384   MLP gate / up      (36 calls)
    M=512, K=16384, N=2048    MLP down           (18 calls)

  SigLIP (M=256, head_dim padded 72→96, so QKV proj fan-out differs):
    M=256, K=1152, N=1152     attn projections   (~108 calls = 27 layers × 4)
    M=256, K=1152, N=4608     FC1 (MLP up)       (27 calls)
    M=256, K=4608, N=1152     FC2 (MLP down)     (27 calls)
    M=256, K=1536, N=1152     attn out (head_dim=96, num_heads=16=1536)

Run:
    PI0_PRE_MM_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_prefill_matmul_sweep.py

Pin shape:
    PI0_PRE_MM_SWEEP_SHAPE=vlm_mlp_gate_up
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest
import torch
import ttnn


BENCH_ENABLED = os.environ.get("PI0_PRE_MM_SWEEP") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_PRE_MM_SWEEP=1 to run the prefill/VLM matmul sweep")

NUM_WARMUP = int(os.environ.get("PI0_PRE_MM_SWEEP_WARMUP", "5"))
NUM_ITER = int(os.environ.get("PI0_PRE_MM_SWEEP_ITER", "20"))
PCC_THRESHOLD = float(os.environ.get("PI0_PRE_MM_PCC", "0.9999"))
SHAPE_FILTER = os.environ.get("PI0_PRE_MM_SWEEP_SHAPE", "").lower()


# (label, M, K, N, activation, weight_dtype)
SHAPES: List[Tuple[str, int, int, int, Optional[Tuple], "ttnn.DataType"]] = [
    ("vlm_qkv_fused", 512, 2048, 2560, None, ttnn.bfloat8_b),
    ("vlm_o_proj", 512, 2048, 2048, None, ttnn.bfloat8_b),
    ("vlm_mlp_gate_up", 512, 2048, 16384, None, ttnn.bfloat8_b),
    ("vlm_mlp_down", 512, 16384, 2048, None, ttnn.bfloat8_b),
    ("siglip_attn_proj", 256, 1152, 1152, None, ttnn.bfloat8_b),
    ("siglip_attn_out", 256, 1536, 1152, None, ttnn.bfloat8_b),
    ("siglip_mlp_fc1", 256, 1152, 4608, None, ttnn.bfloat8_b),
    ("siglip_mlp_fc2", 256, 4608, 1152, None, ttnn.bfloat8_b),
]


@dataclass
class CandConfig:
    in0_block_w: int
    grid_x: int
    grid_y: int
    dst_budget: int
    fp32_dest_acc: bool
    math_fidelity: "ttnn.MathFidelity"
    transpose_mcast: bool = False
    packer_l1_acc: bool = True


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


def _build_2d_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    cfg: CandConfig,
    activation,
) -> Optional["ttnn.MatmulMultiCoreReuseMultiCastProgramConfig"]:
    if k_tiles % cfg.in0_block_w != 0:
        return None
    per_core_M = (m_tiles + cfg.grid_y - 1) // cfg.grid_y
    per_core_N = (n_tiles + cfg.grid_x - 1) // cfg.grid_x
    if per_core_M == 0 or per_core_N == 0:
        return None
    eff_budget = cfg.dst_budget
    sub_w = min(per_core_N, eff_budget)
    while sub_w > 1 and per_core_N % sub_w != 0:
        sub_w -= 1
    sub_h = max(1, eff_budget // sub_w)
    sub_h = min(per_core_M, sub_h)
    while sub_h > 1 and per_core_M % sub_h != 0:
        sub_h -= 1
    try:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(cfg.grid_x, cfg.grid_y),
            in0_block_w=cfg.in0_block_w,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=cfg.transpose_mcast,
            fused_activation=activation,
            fuse_batch=True,
        )
    except Exception:
        return None


def _time_matmul(device, a_tt, w_tt, pcfg, ck_cfg) -> Tuple[float, float]:
    for _ in range(NUM_WARMUP):
        out = ttnn.linear(
            a_tt,
            w_tt,
            program_config=pcfg,
            compute_kernel_config=ck_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = ttnn.linear(
            a_tt,
            w_tt,
            program_config=pcfg,
            compute_kernel_config=ck_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(out)
    return statistics.mean(samples) * 1000, min(samples) * 1000  # µs


def _run_pcc(device, a_tt, w_tt, pcfg, ck_cfg, a_torch, w_torch) -> float:
    out = ttnn.linear(
        a_tt,
        w_tt,
        program_config=pcfg,
        compute_kernel_config=ck_cfg,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out_torch = ttnn.to_torch(out)
    ttnn.deallocate(out)
    ref = a_torch @ w_torch.T
    return _pcc(ref, out_torch.reshape(ref.shape))


def _sweep_one_shape(device, label, M, K, N, activation, weight_dtype):
    print(f"\n{'=' * 80}")
    print(f"  SWEEP: {label}   M={M}  K={K}  N={N}   PCC≥{PCC_THRESHOLD}")
    print(f"{'=' * 80}")

    m_tiles = M // 32
    k_tiles = K // 32
    n_tiles = N // 32

    torch.manual_seed(0)
    a_torch = torch.randn(1, 1, M, K) * 0.1
    w_torch = torch.randn(N, K) * 0.02
    a_tt = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w_tt = ttnn.from_torch(
        w_torch.T.contiguous(),
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Baseline: production picker
    from models.experimental.pi0_5.tt.ttnn_gemma import build_matmul_pcfg as _picker

    baseline_pcfg = _picker(m_tiles, k_tiles, n_tiles, 12, 10, activation=activation)
    baseline_ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    if baseline_pcfg is not None:
        try:
            baseline_pcc = _run_pcc(device, a_tt, w_tt, baseline_pcfg, baseline_ck, a_torch, w_torch)
            baseline_mean, baseline_min = _time_matmul(device, a_tt, w_tt, baseline_pcfg, baseline_ck)
            print(
                f"  Baseline (production picker): mean={baseline_mean:.2f} µs  min={baseline_min:.2f}  pcc={baseline_pcc:.5f}"
            )
        except RuntimeError as e:
            baseline_mean = None
            print(f"  Baseline FAILED: {str(e).splitlines()[0][:120]}")
    else:
        baseline_mean = None
        print(f"  Baseline: picker returned None")

    candidates: List[CandConfig] = []
    block_ws = [bw for bw in (2, 4, 8, 16, 32) if k_tiles % bw == 0]
    grid_choices = [(12, 10), (12, 8), (8, 8), (8, 4)]
    fidelities = [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
    for fid in fidelities:
        for fp32_dest in (False, True):
            for gx, gy in grid_choices:
                for bw in block_ws:
                    for transpose in (False, True):
                        candidates.append(
                            CandConfig(
                                in0_block_w=bw,
                                grid_x=gx,
                                grid_y=gy,
                                dst_budget=4 if fp32_dest else 8,
                                fp32_dest_acc=fp32_dest,
                                math_fidelity=fid,
                                transpose_mcast=transpose,
                                packer_l1_acc=True,
                            )
                        )

    results: List[Tuple[CandConfig, float, float, float]] = []
    for c in candidates:
        pcfg = _build_2d_pcfg(m_tiles, k_tiles, n_tiles, c, activation)
        if pcfg is None:
            continue
        ck_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=c.math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=c.fp32_dest_acc,
            packer_l1_acc=c.packer_l1_acc,
        )
        try:
            pcc = _run_pcc(device, a_tt, w_tt, pcfg, ck_cfg, a_torch, w_torch)
        except RuntimeError:
            continue
        try:
            mean_us, min_us = _time_matmul(device, a_tt, w_tt, pcfg, ck_cfg)
        except RuntimeError:
            continue
        results.append((c, mean_us, min_us, pcc))

    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)

    good = [r for r in results if r[3] >= PCC_THRESHOLD]
    good.sort(key=lambda r: r[1])

    print(f"  Tried: {len(results)}   Passing PCC≥{PCC_THRESHOLD}: {len(good)}")
    print()
    print(
        f"  {'rank':<4}  {'mean µs':>9}  {'min µs':>9}  {'pcc':>8}  {'vs base':>8}  {'bw':>3}  {'grid':>7}  {'dst':>3}  {'fp32':>4}  {'tm':>2}  {'fid':>5}"
    )
    for rank, (c, mean, mn, pcc) in enumerate(good[:20], 1):
        fid = (
            "LoFi"
            if c.math_fidelity == ttnn.MathFidelity.LoFi
            else "HiFi2"
            if c.math_fidelity == ttnn.MathFidelity.HiFi2
            else "HiFi4"
        )
        delta = f"{(mean - baseline_mean):+6.2f}" if baseline_mean is not None else "  n/a"
        print(
            f"  {rank:<4}  {mean:>9.2f}  {mn:>9.2f}  {pcc:>8.5f}  {delta:>8}  {c.in0_block_w:>3}  {c.grid_x:>2}×{c.grid_y:<3}  {c.dst_budget:>3}  {int(c.fp32_dest_acc):>4}  {int(c.transpose_mcast):>2}  {fid:>5}"
        )
    if len(good) > 20:
        print(f"  ... and {len(good) - 20} more")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_prefill_matmul_sweep(device):
    print(f"\nPrefill/VLM/SigLIP matmul sweep — PCC threshold {PCC_THRESHOLD}")
    print(f"warmup={NUM_WARMUP} iter={NUM_ITER}")
    if SHAPE_FILTER:
        print(f"shape filter: {SHAPE_FILTER}")

    for label, M, K, N, activation, wt_dtype in SHAPES:
        if SHAPE_FILTER and SHAPE_FILTER != label:
            continue
        _sweep_one_shape(device, label, M, K, N, activation, wt_dtype)
