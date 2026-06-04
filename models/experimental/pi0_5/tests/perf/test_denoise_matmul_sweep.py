# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep matmul program configs at the pi0.5 denoise-step shapes.

Measures kernel time AND PCC vs torch reference for every (config, shape)
combo, so we only keep configs that preserve numerics.

Denoise matmul shapes (M=32 = 1 tile row):
  QKV fused    : M=32, K=1024, N=2560   (180 calls × 9.11 µs = 1.64 ms total)
  MLP gate/up  : M=32, K=1024, N=4096   (360 calls × 12.51 µs = 4.50 ms)
  O proj       : M=32, K=2048, N=1024   (180 calls × 9.32 µs = 1.68 ms)
  MLP down     : M=32, K=4096, N=1024   (180 calls × 14.40 µs = 2.59 ms)

All four use the 1D width-shard path
(`MatmulMultiCoreReuseMultiCast1DProgramConfig`) since m_tiles=1 makes
the 2D grid mostly empty. The 1D path spreads N across all 120 cores.

Sweep axes (in priority order):
  - in0_block_w         {2, 4, 8, 16, 32}  must divide K_tiles
  - num_cores           {24, 48, 60, 96, 120}  controls per_core_N
  - dst_budget          {4, 8}  with matching fp32_dest_acc
  - fp32_dest_acc_en    {False, True}
  - math_fidelity       {LoFi, HiFi2, HiFi4}

Each config is gated: skipped if it would require chunks that don't
divide K or N; rejected from the leaderboard if PCC vs torch < 0.999.

Run:
    PI0_MM_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_denoise_matmul_sweep.py

Pin to one shape:
    PI0_MM_SWEEP_SHAPE=mlp_gate_up

PCC threshold (default 0.999):
    PI0_MM_PCC=0.999
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


BENCH_ENABLED = os.environ.get("PI0_MM_SWEEP") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_MM_SWEEP=1 to run the denoise matmul sweep")

NUM_WARMUP = int(os.environ.get("PI0_MM_SWEEP_WARMUP", "5"))
NUM_ITER = int(os.environ.get("PI0_MM_SWEEP_ITER", "30"))
PCC_THRESHOLD = float(os.environ.get("PI0_MM_PCC", "0.999"))
SHAPE_FILTER = os.environ.get("PI0_MM_SWEEP_SHAPE", "").lower()  # blank = all


# (label, M, K, N, activation)
SHAPES: List[Tuple[str, int, int, int, Optional[Tuple]]] = [
    ("qkv_fused", 32, 1024, 2560, None),
    ("mlp_gate_up", 32, 1024, 4096, None),
    ("o_proj", 32, 2048, 1024, None),
    ("mlp_down", 32, 4096, 1024, None),
]


@dataclass
class CandConfig:
    in0_block_w: int
    num_cores: int
    dst_budget: int
    fp32_dest_acc: bool
    math_fidelity: "ttnn.MathFidelity"
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


def _build_1d_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    cfg: CandConfig,
    activation,
) -> Optional["ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig"]:
    """Try to build a 1D width-shard matmul program config. Returns None if
    the shape/cores combo is invalid."""
    if k_tiles % cfg.in0_block_w != 0:
        return None
    if cfg.num_cores > 120:
        return None
    if n_tiles % cfg.num_cores != 0:
        # Allow ceil division; some kernels accept it
        per_core_N = (n_tiles + cfg.num_cores - 1) // cfg.num_cores
    else:
        per_core_N = n_tiles // cfg.num_cores
    if per_core_N == 0:
        return None
    # subblock w/h from dst_budget. With fp32_dest_acc the kernel halves it.
    eff_budget = cfg.dst_budget
    sub_w = min(per_core_N, eff_budget)
    while sub_w > 1 and per_core_N % sub_w != 0:
        sub_w -= 1
    sub_h = max(1, eff_budget // sub_w)
    sub_h = min(m_tiles, sub_h)
    while sub_h > 1 and m_tiles % sub_h != 0:
        sub_h -= 1
    # grid shape — 12 wide, ceil(num_cores/12) tall.
    cfg_gx = min(12, cfg.num_cores)
    cfg_gy = (cfg.num_cores + cfg_gx - 1) // cfg_gx
    try:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cfg_gx, cfg_gy),
            in0_block_w=cfg.in0_block_w,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=m_tiles,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=activation,
            mcast_in0=True,
        )
    except Exception:
        return None


def _time_matmul(
    device,
    a_tt: "ttnn.Tensor",
    w_tt: "ttnn.Tensor",
    pcfg,
    ck_cfg,
) -> Tuple[float, float]:
    """Returns (mean_ms, min_ms) over NUM_ITER iters."""
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
    """Compute one output, compare to torch reference, return PCC."""
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
    # Torch reference: a @ w.T (ttnn.linear matches torch.nn.functional.linear)
    ref = a_torch @ w_torch.T
    return _pcc(ref, out_torch.reshape(ref.shape))


def _sweep_one_shape(device, label, M, K, N, activation):
    print(f"\n{'=' * 80}")
    print(f"  SWEEP: {label}   M={M}  K={K}  N={N}   PCC threshold={PCC_THRESHOLD}")
    print(f"{'=' * 80}")

    m_tiles = M // 32
    k_tiles = K // 32
    n_tiles = N // 32

    # Build inputs once
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
    # ttnn.linear: weight is [K, N] (transposed). Upload as .T to match.
    w_tt = ttnn.from_torch(
        w_torch.T.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # ---- Baseline: production picker's choice (build_matmul_pcfg). ----
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
        print(f"  Baseline: picker returned None for this shape")

    # Build candidate configs. Include cores=64 (production picker's pick for N=128)
    # and 32, 80 as additional points around the heuristic's choices.
    candidates: List[CandConfig] = []
    block_ws = [bw for bw in (2, 4, 8, 16, 32) if k_tiles % bw == 0]
    core_choices = [24, 32, 48, 60, 64, 80, 96, 120]
    fidelities = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
    for fid in fidelities:
        for fp32_dest in (False, True):
            for nc in core_choices:
                for bw in block_ws:
                    candidates.append(
                        CandConfig(
                            in0_block_w=bw,
                            num_cores=nc,
                            dst_budget=4 if fp32_dest else 8,
                            fp32_dest_acc=fp32_dest,
                            math_fidelity=fid,
                            packer_l1_acc=True,
                        )
                    )

    results: List[Tuple[CandConfig, float, float, float]] = []
    for c in candidates:
        pcfg = _build_1d_pcfg(m_tiles, k_tiles, n_tiles, c, activation)
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
        except RuntimeError as e:
            tag = f"bw={c.in0_block_w:>3}  nc={c.num_cores:>4}  fid={c.math_fidelity!s:18}  fp32_dest={int(c.fp32_dest_acc)}"
            print(f"   FAIL {tag}: {str(e).splitlines()[0][:80]}")
            continue
        try:
            mean_us, min_us = _time_matmul(device, a_tt, w_tt, pcfg, ck_cfg)
        except RuntimeError as e:
            print(f"   TIME-FAIL bw={c.in0_block_w} nc={c.num_cores}: {str(e).splitlines()[0][:80]}")
            continue
        results.append((c, mean_us, min_us, pcc))

    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)

    # Filter by PCC; report
    good = [r for r in results if r[3] >= PCC_THRESHOLD]
    good.sort(key=lambda r: r[1])  # by mean time ASC

    print(f"  Total tried: {len(results)}   Passing PCC≥{PCC_THRESHOLD}: {len(good)}")
    print()
    print(
        f"  {'rank':<4}  {'mean µs':>9}  {'min µs':>9}  {'pcc':>8}  {'vs base':>8}  {'bw':>3}  {'cores':>5}  {'dst':>3}  {'fp32':>4}  {'fidelity':>7}"
    )
    for rank, (c, mean, mn, pcc) in enumerate(good[:25], 1):
        fid = (
            "LoFi"
            if c.math_fidelity == ttnn.MathFidelity.LoFi
            else "HiFi2"
            if c.math_fidelity == ttnn.MathFidelity.HiFi2
            else "HiFi4"
        )
        delta_str = f"{(mean - baseline_mean):+6.2f}" if baseline_mean is not None else "  n/a"
        print(
            f"  {rank:<4}  {mean:>9.2f}  {mn:>9.2f}  {pcc:>8.5f}  {delta_str:>8}  {c.in0_block_w:>3}  {c.num_cores:>5}  {c.dst_budget:>3}  {int(c.fp32_dest_acc):>4}  {fid:>7}"
        )
    if len(good) > 25:
        print(f"   ... and {len(good) - 25} more")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_matmul_sweep(device):
    print(f"\nDenoise matmul sweep — PCC threshold {PCC_THRESHOLD}")
    print(f"warmup={NUM_WARMUP} iter={NUM_ITER}")
    if SHAPE_FILTER:
        print(f"shape filter: {SHAPE_FILTER}")

    for label, M, K, N, activation in SHAPES:
        if SHAPE_FILTER and SHAPE_FILTER != label:
            continue
        _sweep_one_shape(device, label, M, K, N, activation)
