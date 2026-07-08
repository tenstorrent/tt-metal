# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive matmul sharding sweep for pi0.5 production shapes.

For each shape in pi0.5's matmul catalog (see project_pi05_matmul_shape_catalog
memory), iterate the cross-product of:

  - Program-config variant:
      * 1D width-shard mcast  (MatmulMultiCoreReuseMultiCast1DProgramConfig)
      * 2D block-shard mcast  (MatmulMultiCoreReuseMultiCastProgramConfig)
      (manual BMM-reuse omitted — not used in pi0.5 matmuls outside attn)

  - num_cores  ∈ divisors of N_tiles (for 1D) or grid factorisations (for 2D)
  - in0_block_w  ∈ divisors of K_tiles, candidates {4, 8, 16, 32, 64, K_tiles}
  - subblock     ∈ {(1,1) ... (8,1)} with h*w ≤ 8 and h ≤ M_tiles

Locked compute config (per PERF_PLAYBOOKS/05 §9):
  - bf8b weights, bf16 activation (model dtype)
  - LoFi math fidelity
  - fp32_dest_acc_en = False  (unlocks subblock cap 4→8 — 01 §3)
  - packer_l1_acc = True      (mandatory — else 3.5× wrong times, 07 §4)

Per-candidate gate: PCC ≥ threshold vs torch reference; configs failing
are dropped from the leaderboard.

Timing: median-of-N (default 30 iters, 10 warmup). Wall-clock with
synchronize_device brackets is unreliable below ~50 µs/call — the
playbook-noted dispatch floor. Cross-verify the top 2-3 candidates with
tracy on the production e2e test before committing.

Run one shape:
    PI0_MM_SHARDING_SWEEP=1 PI0_MM_SHAPE=denoise_mlp_gate_up \\
      pytest -xvs models/experimental/pi0_5/tests/perf/test_matmul_sharding_sweep.py

Available shapes (see project_pi05_matmul_shape_catalog):
    denoise_mlp_gate_up     M=32  K=1024 N=4096   1D  360× 4.50 ms (biggest 1D)
    denoise_mlp_down        M=32  K=4096 N=1024   1D  180× 2.45 ms (16 cores)
    denoise_qkv_fused       M=32  K=1024 N=2560   1D  180× 1.63 ms
    denoise_o_proj          M=32  K=2048 N=1024   1D  180× 1.47 ms (PI0_DENOISE_MM_TUNE)
    vlm_mlp_gate            M=512 K=2048 N=16384  2D   36× 5.48 ms (biggest 2D)
    vlm_mlp_down            M=512 K=16384 N=2048  2D   18× 2.95 ms (PI0_PREFILL_MM_TUNE)
    vlm_qkv_fused           M=512 K=2048 N=2560   2D   18× 0.51 ms
    vlm_o_proj              M=512 K=2048 N=2048   2D   18× 0.46 ms
    siglip_mlp_gate         M=256 K=1152 N=4608   2D   54× 1.53 ms
    siglip_mlp_down         M=256 K=4608 N=1152   2D   27× 1.11 ms
    siglip_qkv_fused        M=256 K=1536 N=1152   2D   27× 0.39 ms

Output CSV: writes leaderboard to /tmp/pi05_mm_sharding_sweep_<SHAPE>.csv
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch
import ttnn


BENCH_ENABLED = os.environ.get("PI0_MM_SHARDING_SWEEP") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_MM_SHARDING_SWEEP=1 to run the matmul sharding sweep",
)

# Per-shape config: (label, M, K, N, activation)
# Activation tuple: (ttnn.UnaryOpType, approx_mode_bool) or None
SHAPES = {
    "denoise_mlp_gate_up": (32, 1024, 4096, None),
    "denoise_mlp_down": (32, 4096, 1024, None),
    "denoise_qkv_fused": (32, 1024, 2560, None),
    "denoise_o_proj": (32, 2048, 1024, None),
    "vlm_mlp_gate": (512, 2048, 16384, None),
    "vlm_mlp_down": (512, 16384, 2048, None),
    "vlm_qkv_fused": (512, 2048, 2560, None),
    "vlm_o_proj": (512, 2048, 2048, None),
    "siglip_mlp_gate": (256, 1152, 4608, None),
    "siglip_mlp_down": (256, 4608, 1152, None),
    "siglip_qkv_fused": (256, 1536, 1152, None),
}

NUM_WARMUP = int(os.environ.get("PI0_MM_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_MM_ITER", "30"))
PCC_THRESHOLD = float(os.environ.get("PI0_MM_PCC", "0.999"))
SHAPE_LABEL = os.environ.get("PI0_MM_SHAPE", "denoise_mlp_gate_up")
OUT_DIR = os.environ.get("PI0_MM_OUT", "/tmp")


@dataclass
class CandResult:
    pcfg_kind: str  # "1d_mcast" or "2d_block"
    num_cores: int
    grid_x: int
    grid_y: int
    in0_block_w: int
    subblock_h: int
    subblock_w: int
    per_core_M: int
    per_core_N: int
    mean_us: float
    median_us: float
    min_us: float
    pcc: float
    error: str = ""


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


def _divisors(n: int) -> List[int]:
    out = []
    for i in range(1, n + 1):
        if n % i == 0:
            out.append(i)
    return out


def _subblock_candidates(m_tiles: int, per_core_N_tiles: int, cap: int = 8) -> List[Tuple[int, int]]:
    """Subblock (h, w) candidates with h*w <= cap, h <= m_tiles, w <= per_core_N."""
    out = []
    for h in range(1, min(m_tiles, cap) + 1):
        if m_tiles % h != 0:
            continue
        for w in range(1, min(per_core_N_tiles, cap // h) + 1):
            if per_core_N_tiles % w != 0:
                continue
            out.append((h, w))
    return out


def _build_1d_pcfg(m_t, k_t, n_t, num_cores, ibw, sub_h, sub_w, activation):
    if num_cores < 1 or num_cores > 120:
        return None, None, None
    if k_t % ibw != 0:
        return None, None, None
    per_core_N = (n_t + num_cores - 1) // num_cores
    if per_core_N == 0:
        return None, None, None
    if per_core_N % sub_w != 0:
        return None, None, None
    if m_t % sub_h != 0:
        return None, None, None
    cfg_gx = min(12, num_cores)
    cfg_gy = (num_cores + cfg_gx - 1) // cfg_gx
    try:
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cfg_gx, cfg_gy),
            in0_block_w=ibw,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=m_t,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=activation,
            mcast_in0=True,
        )
        return cfg, (cfg_gx, cfg_gy), per_core_N
    except Exception:
        return None, None, None


def _build_2d_pcfg(m_t, k_t, n_t, grid_x, grid_y, ibw, sub_h, sub_w, activation):
    if grid_x < 1 or grid_y < 1 or grid_x * grid_y > 120:
        return None, None, None
    if grid_x > 12 or grid_y > 10:
        return None, None, None
    if m_t % grid_y != 0 or n_t % grid_x != 0:
        return None, None, None
    if k_t % ibw != 0:
        return None, None, None
    per_core_M = m_t // grid_y
    per_core_N = n_t // grid_x
    if per_core_M % sub_h != 0 or per_core_N % sub_w != 0:
        return None, None, None
    try:
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=ibw,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=activation,
        )
        return cfg, (grid_x, grid_y), per_core_N
    except Exception:
        return None, None, None


def _time_matmul(device, a_tt, w_tt, pcfg, ck_cfg) -> Tuple[float, float, float]:
    """Returns (mean_us, median_us, min_us)."""
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
        samples.append((time.perf_counter() - t0) * 1_000_000)  # µs
        ttnn.deallocate(out)
    return statistics.mean(samples), statistics.median(samples), min(samples)


def _candidate_pcc(device, a_tt, w_tt, pcfg, ck_cfg, a_torch, w_torch) -> float:
    try:
        out = ttnn.linear(
            a_tt,
            w_tt,
            program_config=pcfg,
            compute_kernel_config=ck_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ref_out = ttnn.to_torch(out).reshape(1, 1, a_torch.shape[-2], w_torch.shape[-1])
        ttnn.deallocate(out)
    except RuntimeError as e:
        return -1.0
    torch_ref = (a_torch.float() @ w_torch.float()).unsqueeze(0).unsqueeze(0)
    return _pcc(torch_ref, ref_out.float())


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_matmul_sharding_sweep(device):
    if SHAPE_LABEL not in SHAPES:
        pytest.fail(f"Unknown PI0_MM_SHAPE={SHAPE_LABEL}; choose from {list(SHAPES.keys())}")
    M, K, N, activation = SHAPES[SHAPE_LABEL]
    m_t, k_t, n_t = M // 32, K // 32, N // 32
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y

    print()
    print("=" * 90)
    print(f"  SWEEP: {SHAPE_LABEL}   M={M}  K={K}  N={N}")
    print(f"    M_tiles={m_t}  K_tiles={k_t}  N_tiles={n_t}")
    print(f"    grid={grid.x}×{grid.y} ({total_cores} cores)")
    print(f"    warmup={NUM_WARMUP}  iter={NUM_ITER}  PCC≥{PCC_THRESHOLD}")
    print("=" * 90)

    # Inputs (CPU side) + upload once
    torch.manual_seed(0)
    a_torch = torch.randn(M, K) * 0.1
    w_torch = torch.randn(K, N) * 0.1
    a_tt = ttnn.from_torch(
        a_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w_tt = ttnn.from_torch(
        w_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compute kernel — locked per playbook 05 §9
    ck_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # unlocks subblock h*w ≤ 8
        packer_l1_acc=True,
    )

    candidates: List[Tuple[str, dict]] = []

    # 1D width-shard candidates (mcast_in0=True): vary num_cores ∈ {16,24,32,48,60,64,80,96,120}
    for num_cores in (16, 24, 32, 48, 60, 64, 80, 96, 120):
        per_core_N = (n_t + num_cores - 1) // num_cores
        if per_core_N == 0:
            continue
        # in0_block_w candidates: divisors of K_tiles, capped
        for ibw in (4, 8, 16, 32, 64, k_t):
            if ibw > k_t or k_t % ibw != 0:
                continue
            for sub_h, sub_w in _subblock_candidates(m_t, per_core_N, cap=8):
                candidates.append(
                    (
                        "1d_mcast",
                        dict(
                            num_cores=num_cores,
                            ibw=ibw,
                            sub_h=sub_h,
                            sub_w=sub_w,
                        ),
                    )
                )

    # 2D block-shard candidates: grid factorisations of (grid_x ≤ 12, grid_y ≤ 10)
    # where grid_y | M_tiles and grid_x | N_tiles
    for grid_y in _divisors(min(m_t, 10)):
        for grid_x in _divisors(min(n_t, 12)):
            num_cores = grid_x * grid_y
            if num_cores > total_cores:
                continue
            if num_cores < 4:  # waste of grid
                continue
            per_core_N = n_t // grid_x
            per_core_M = m_t // grid_y
            for ibw in (4, 8, 16, 32, 64, k_t):
                if ibw > k_t or k_t % ibw != 0:
                    continue
                for sub_h, sub_w in _subblock_candidates(per_core_M, per_core_N, cap=8):
                    candidates.append(
                        (
                            "2d_block",
                            dict(
                                grid_x=grid_x,
                                grid_y=grid_y,
                                ibw=ibw,
                                sub_h=sub_h,
                                sub_w=sub_w,
                            ),
                        )
                    )

    # Dedupe
    seen = set()
    deduped = []
    for k, p in candidates:
        key = (k, tuple(sorted(p.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((k, p))
    candidates = deduped

    print(f"\n  {len(candidates)} candidate configs")
    print()

    # Sweep
    results: List[CandResult] = []
    for i, (kind, p) in enumerate(candidates):
        if kind == "1d_mcast":
            pcfg, grid_wh, per_core_N = _build_1d_pcfg(
                m_t,
                k_t,
                n_t,
                p["num_cores"],
                p["ibw"],
                p["sub_h"],
                p["sub_w"],
                activation,
            )
            if pcfg is None:
                continue
            num_cores = p["num_cores"]
            gx, gy = grid_wh
            per_core_M = m_t
        else:
            pcfg, grid_wh, per_core_N = _build_2d_pcfg(
                m_t,
                k_t,
                n_t,
                p["grid_x"],
                p["grid_y"],
                p["ibw"],
                p["sub_h"],
                p["sub_w"],
                activation,
            )
            if pcfg is None:
                continue
            gx, gy = grid_wh
            num_cores = gx * gy
            per_core_M = m_t // gy

        # PCC gate
        pcc = _candidate_pcc(device, a_tt, w_tt, pcfg, ck_cfg, a_torch, w_torch)
        if pcc < PCC_THRESHOLD:
            results.append(
                CandResult(
                    kind,
                    num_cores,
                    gx,
                    gy,
                    p["ibw"],
                    p["sub_h"],
                    p["sub_w"],
                    per_core_M,
                    per_core_N,
                    0,
                    0,
                    0,
                    pcc,
                    "pcc_fail",
                )
            )
            continue

        # Time
        try:
            mean_us, median_us, min_us = _time_matmul(device, a_tt, w_tt, pcfg, ck_cfg)
        except RuntimeError as e:
            results.append(
                CandResult(
                    kind,
                    num_cores,
                    gx,
                    gy,
                    p["ibw"],
                    p["sub_h"],
                    p["sub_w"],
                    per_core_M,
                    per_core_N,
                    0,
                    0,
                    0,
                    pcc,
                    f"timing_err",
                )
            )
            continue
        results.append(
            CandResult(
                kind,
                num_cores,
                gx,
                gy,
                p["ibw"],
                p["sub_h"],
                p["sub_w"],
                per_core_M,
                per_core_N,
                mean_us,
                median_us,
                min_us,
                pcc,
            )
        )

    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)

    # Sort and print top-15 (by median)
    valid = [r for r in results if r.pcc >= PCC_THRESHOLD]
    valid.sort(key=lambda r: r.median_us)

    print(f"  Tried: {len(results)}   Passing PCC≥{PCC_THRESHOLD}: {len(valid)}")
    print()
    print(
        f"  {'rank':<4} {'kind':<10} {'cores':>5} {'gx':>3} {'gy':>3} {'ibw':>4} {'sub_h':>5} {'sub_w':>5} {'pc_M':>4} {'pc_N':>4}  {'mean µs':>9} {'med µs':>8} {'min µs':>8}  {'pcc':>7}"
    )
    for rank, r in enumerate(valid[:15], 1):
        print(
            f"  {rank:<4} {r.pcfg_kind:<10} {r.num_cores:>5} {r.grid_x:>3} {r.grid_y:>3} {r.in0_block_w:>4} {r.subblock_h:>5} {r.subblock_w:>5} {r.per_core_M:>4} {r.per_core_N:>4}  {r.mean_us:>9.2f} {r.median_us:>8.2f} {r.min_us:>8.2f}  {r.pcc:>7.5f}"
        )

    # Save CSV
    out_path = os.path.join(OUT_DIR, f"pi05_mm_sharding_sweep_{SHAPE_LABEL}.csv")
    with open(out_path, "w") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "rank",
                "kind",
                "cores",
                "gx",
                "gy",
                "ibw",
                "sub_h",
                "sub_w",
                "pc_M",
                "pc_N",
                "mean_us",
                "median_us",
                "min_us",
                "pcc",
                "error",
            ]
        )
        for rank, r in enumerate(valid, 1):
            wr.writerow(
                [
                    rank,
                    r.pcfg_kind,
                    r.num_cores,
                    r.grid_x,
                    r.grid_y,
                    r.in0_block_w,
                    r.subblock_h,
                    r.subblock_w,
                    r.per_core_M,
                    r.per_core_N,
                    f"{r.mean_us:.2f}",
                    f"{r.median_us:.2f}",
                    f"{r.min_us:.2f}",
                    f"{r.pcc:.5f}",
                    "",
                ]
            )
        for r in results:
            if r.error:
                wr.writerow(
                    [
                        "err",
                        r.pcfg_kind,
                        r.num_cores,
                        r.grid_x,
                        r.grid_y,
                        r.in0_block_w,
                        r.subblock_h,
                        r.subblock_w,
                        r.per_core_M,
                        r.per_core_N,
                        "",
                        "",
                        "",
                        f"{r.pcc:.5f}",
                        r.error,
                    ]
                )
    print(f"\n  CSV saved: {out_path}")
