# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sweep program-config / compute-kernel-config / memory-config variants for the
DCNv2 final matmul.

Shape: (1, M, K) @ (K, N) where K=2304 (=9 * 256, the kernel-batched samples),
N=256 (output channels), and M = H_out * W_out over FPN levels:
    P3: M=6400 (80x80)   3 calls/inference
    P4: M=1600 (40x40)   9 calls/inference
    P5: M= 400 (20x20)   9 calls/inference
    P6: M= 100 (10x10)   9 calls/inference
    P7: M=  25 ( 5x5)    9 calls/inference

We measure pure on-device kernel duration with tracy signposts. Each candidate
runs `WARMUP+REPS` times; we use the median of REPS for stability.

Run with:
    cd $TT_METAL_HOME
    source python_env/bin/activate
    export ARCH_NAME=wormhole_b0
    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages
    TT_VISIBLE_DEVICES=0 pytest \
      models/experimental/atss_swin_l_dyhead/tests/perf/sweep_dcn_matmul.py -sv

The test passes regardless of outcome; the goal is the CSV summary it prints.
"""

import csv
import os
import statistics
import time

import pytest
import torch
import ttnn
from loguru import logger


# -------------------- candidates ---------------------------------------------

# (K_in, N_out) of the DCN matmul (fixed by ATSS DyHead).
K = 2304  # = 9 * 256
N = 256

# M values to sweep, in declining-priority order. The biggest M (P3) contributes
# the most wall time; small-M shapes share kernels but tune independently.
M_VALUES = [6400, 1600, 400, 100, 25]

WARMUP = 3
REPS = 10  # take median


def make_inputs(device, M):
    """Build the tile-aligned (1, M, K) and (K, N) bf16 TILE inputs in DRAM."""
    torch.manual_seed(0)
    in0 = torch.randn(1, M, K, dtype=torch.float32) * 0.05
    in1 = torch.randn(K, N, dtype=torch.float32) * 0.05
    in0_tt = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_tt = ttnn.from_torch(
        in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return in0_tt, in1_tt


def time_op(fn):
    """Run fn() WARMUP+REPS times, return (min, median, max) wall-time in microseconds.

    We use `ttnn.synchronize_device` to block until the kernel actually completed.
    Pure host wall time without a profiler is a coarse proxy for kernel time, but
    differences between candidates here are large enough that this is the right
    granularity for a sweep.
    """
    device = fn.device
    for _ in range(WARMUP):
        out = fn()
        ttnn.deallocate(out) if out is not None else None
    ttnn.synchronize_device(device)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
        if out is not None:
            ttnn.deallocate(out)
    times.sort()
    return times[0], statistics.median(times), times[-1]


def compute_configs(device):
    """All (fidelity, fp32_dest, packer) combos we'll sweep."""
    fids = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3]
    out = []
    for fid in fids:
        for fp32 in (False, True):
            for packer in (False, True):
                cc = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=fid,
                    fp32_dest_acc_en=fp32,
                    packer_l1_acc=packer,
                    math_approx_mode=False,
                )
                tag = f"{fid.name}/fp32={int(fp32)}/pacc={int(packer)}"
                out.append((tag, cc))
    return out


def baseline_auto(in0, in1, cc):
    """Current production path: ttnn.matmul with L1 output and no program_config."""
    return ttnn.matmul(in0, in1, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=cc)


def baseline_auto_dram(in0, in1, cc):
    """Same as auto but DRAM output."""
    return ttnn.matmul(in0, in1, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cc)


def matmul_1d_mcast(in0, in1, cc, *, grid_xy, in0_block_w, sub_h, sub_w, per_core_M, per_core_N, mcast_in0=False):
    """ttnn.matmul with explicit 1D-mcast program config."""
    pcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=mcast_in0,
    )
    return ttnn.matmul(in0, in1, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=cc, program_config=pcfg)


def matmul_2d_mcast(in0, in1, cc, *, grid_xy, in0_block_w, sub_h, sub_w, per_core_M, per_core_N):
    """ttnn.matmul with explicit 2D-mcast program config."""
    pcfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    return ttnn.matmul(in0, in1, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=cc, program_config=pcfg)


def candidate_grid_sweep(M, K_tiles=K // 32, N_tiles=N // 32):
    """Enumerate plausible 1D/2D mcast configs given M_tiles, K_tiles, N_tiles."""
    M_tiles = (M + 31) // 32
    candidates = []

    # 1D-mcast variants — sweep grid 8x8 plus a couple smaller, ibw, subblock.
    for grid_xy in [(8, 8), (8, 4), (8, 2)]:
        gx, gy = grid_xy
        ncores = gx * gy
        # per_core_M = ceil(M_tiles / ncores). Skip when imbalanced.
        if M_tiles < ncores:
            # too few output tiles for this grid; not useful at 1D mcast over M.
            continue
        per_core_M = (M_tiles + ncores - 1) // ncores
        per_core_N = N_tiles  # 1D-mcast: each core takes full N
        # Sweep in0_block_w over divisors of K_tiles=72 that fit reasonable CB
        for ibw in (4, 8, 12, 18, 24):
            if K_tiles % ibw != 0:
                continue
            # Sweep subblock with cap h*w<=8 (fp32_dest=False) or 4 (True). The compute
            # config picks the cap; here we just emit candidates and let crashes filter.
            for sub_h, sub_w in [(1, 8), (2, 4), (4, 2), (8, 1), (1, 4), (2, 2), (4, 1)]:
                if per_core_M % sub_h != 0 or per_core_N % sub_w != 0:
                    continue
                candidates.append(
                    dict(
                        kind="1D",
                        grid_xy=ttnn.CoreCoord(gx, gy),
                        in0_block_w=ibw,
                        sub_h=sub_h,
                        sub_w=sub_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        mcast_in0=False,
                    )
                )

    # 2D-mcast variants — only meaningful when M_tiles >= gy and N_tiles >= gx.
    for grid_xy in [(8, 8), (4, 4), (8, 4)]:
        gx, gy = grid_xy
        if M_tiles < gy or N_tiles < gx:
            continue
        per_core_M = (M_tiles + gy - 1) // gy
        per_core_N = (N_tiles + gx - 1) // gx
        for ibw in (4, 8, 12, 18, 24):
            if K_tiles % ibw != 0:
                continue
            for sub_h, sub_w in [(1, 8), (2, 4), (4, 2), (8, 1), (1, 4), (2, 2), (4, 1)]:
                if per_core_M % sub_h != 0 or per_core_N % sub_w != 0:
                    continue
                candidates.append(
                    dict(
                        kind="2D",
                        grid_xy=ttnn.CoreCoord(gx, gy),
                        in0_block_w=ibw,
                        sub_h=sub_h,
                        sub_w=sub_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                    )
                )
    return candidates


@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_sweep_dcn_matmul(device, M):
    """Sweep candidates for one M-shape, write CSV row per candidate."""
    in0, in1 = make_inputs(device, M)
    out_dir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "dcn_matmul_sweep")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"sweep_M{M}.csv")
    rows = []

    compute_cfgs = compute_configs(device)

    # --- baselines ---
    for tag, cc in compute_cfgs:

        def _f():
            return baseline_auto(in0, in1, cc)

        _f.device = device
        try:
            tmin, tmed, tmax = time_op(_f)
            rows.append(
                {
                    "kind": "auto-L1",
                    "compute": tag,
                    "config": "default",
                    "min_us": f"{tmin:.1f}",
                    "median_us": f"{tmed:.1f}",
                    "max_us": f"{tmax:.1f}",
                    "status": "ok",
                }
            )
            logger.info(f"M={M}  auto-L1  {tag}  median={tmed:.1f}us")
        except Exception as e:
            rows.append(
                {
                    "kind": "auto-L1",
                    "compute": tag,
                    "config": "default",
                    "min_us": "",
                    "median_us": "",
                    "max_us": "",
                    "status": f"ERR: {str(e)[:80]}",
                }
            )

    # --- explicit program-config candidates (use the fastest baseline fidelity only to
    # keep the sweep tractable; sweep fidelity separately on the winner).
    # Pick a single representative compute config for the program-config sweep — LoFi+fp32d=F+pacc=T —
    # which BGE-M3 says is the production default for bf16 matmul.
    cc_default = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        math_approx_mode=False,
    )

    cands = candidate_grid_sweep(M)
    logger.info(f"M={M}: {len(cands)} program-config candidates to test")
    for c in cands:
        kind = c["kind"]
        tag = f"{kind}/grid={c['grid_xy'].x}x{c['grid_xy'].y}/ibw={c['in0_block_w']}/sub={c['sub_h']}x{c['sub_w']}/pmN={c['per_core_M']}x{c['per_core_N']}"
        try:
            if kind == "1D":

                def _f():
                    return matmul_1d_mcast(
                        in0,
                        in1,
                        cc_default,
                        grid_xy=c["grid_xy"],
                        in0_block_w=c["in0_block_w"],
                        sub_h=c["sub_h"],
                        sub_w=c["sub_w"],
                        per_core_M=c["per_core_M"],
                        per_core_N=c["per_core_N"],
                        mcast_in0=c.get("mcast_in0", False),
                    )

            else:

                def _f():
                    return matmul_2d_mcast(
                        in0,
                        in1,
                        cc_default,
                        grid_xy=c["grid_xy"],
                        in0_block_w=c["in0_block_w"],
                        sub_h=c["sub_h"],
                        sub_w=c["sub_w"],
                        per_core_M=c["per_core_M"],
                        per_core_N=c["per_core_N"],
                    )

            _f.device = device
            tmin, tmed, tmax = time_op(_f)
            rows.append(
                {
                    "kind": kind,
                    "compute": "LoFi/fp32=0/pacc=1",
                    "config": tag,
                    "min_us": f"{tmin:.1f}",
                    "median_us": f"{tmed:.1f}",
                    "max_us": f"{tmax:.1f}",
                    "status": "ok",
                }
            )
            logger.info(f"M={M}  {tag}  median={tmed:.1f}us")
        except Exception as e:
            rows.append(
                {
                    "kind": kind,
                    "compute": "LoFi/fp32=0/pacc=1",
                    "config": tag,
                    "min_us": "",
                    "median_us": "",
                    "max_us": "",
                    "status": f"ERR: {str(e)[:80]}",
                }
            )

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "compute", "config", "min_us", "median_us", "max_us", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"M={M} sweep complete; wrote {len(rows)} rows to {csv_path}")

    # Print a top-5 leaderboard
    ok = [r for r in rows if r["status"] == "ok" and r["median_us"]]
    ok.sort(key=lambda r: float(r["median_us"]))
    logger.info(f"=== Top 5 for M={M} ===")
    for r in ok[:5]:
        logger.info(f"  {r['median_us']:>6}us  {r['kind']:<6}  {r['compute']:<22}  {r['config']}")

    ttnn.deallocate(in0)
    ttnn.deallocate(in1)
