#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2.1.c microbench: measure LoFi peak matmul TFLOPs with and without
# the on-chip ttnvtop sampler enabled, and print the relative regression.
#
# Until the Phase 2.1.c LLK between-tile sampler hooks land in the tt_llk
# submodule, "sampler ON" and "sampler OFF" produce the same numbers — the
# only thing the env var TTNVTOP_REGISTER_PROGRAMS=1 currently toggles is the
# host-side program registrar (a few microseconds of extra dispatch work,
# nothing on the device). This script is the harness; the data becomes
# meaningful when the LLK hooks are instrumented.
#
# Usage:
#   # Run the OFF case (5 warmup + 100 measured iters, no env var)
#   python tt_metal/tools/ttnvtop/scripts/microbench_phase21c.py --sampler-off
#
#   # Run the ON case (env var TTNVTOP_REGISTER_PROGRAMS=1 set)
#   python tt_metal/tools/ttnvtop/scripts/microbench_phase21c.py --sampler-on
#
#   # Run BOTH back-to-back and print regression verdict
#   python tt_metal/tools/ttnvtop/scripts/microbench_phase21c.py --both
#
# Output: prints median / p10 / p90 TFLOPs per case, and (for --both) the
# regression % vs the 1.0% target, with PASS/FAIL verdict. Per-iteration
# measurements are dumped to runs/<ts>/microbench_phase21c.csv.

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Workload parameters: 4096^3 LoFi bf8b matmul on the full 8x8 grid via ETH
# dispatch. Mirrors test_ttnvtop_full_grid_peak_via_eth_dispatch in the
# accuracy suite.
M = 4096
K = 4096
N = 4096
TILE = 32
GRID = (8, 8)
WARMUP_ITERS = 5
MEASURE_ITERS = 100
FLOPS_PER_ITER = 2 * M * K * N
REGRESSION_TARGET_PCT = 1.0


def _run_one_case(label):
    """Open device, run WARMUP_ITERS+MEASURE_ITERS iters, return list of per-iter TFLOPs."""
    # Imports deferred so --help and --both (which spawns subprocesses) don't
    # need ttnn just to print usage.
    import torch
    import ttnn

    device = ttnn.open_device(
        device_id=0,
        l1_small_size=24576,
        trace_region_size=3855488,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )
    try:
        cg = device.compute_with_storage_grid_size()
        if cg.x < GRID[0] or cg.y < GRID[1]:
            print(
                f"[microbench:{label}] WARN: ETH dispatch did not expose full {GRID} grid "
                f"(got {cg.x}x{cg.y}). Numbers will not reflect peak.",
                file=sys.stderr,
            )

        in0 = torch.randn((1, 1, M, K)).bfloat16()
        in1 = torch.randn((1, 1, K, N)).bfloat16()
        in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=GRID,
            in0_block_w=K // GRID[0] // TILE,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=M // GRID[1] // TILE,
            per_core_N=N // GRID[0] // TILE,
            transpose_mcast=False,
            fused_activation=None,
        )
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Warmup (JIT compile, cache priming, EWMA settle).
        for _ in range(WARMUP_ITERS):
            ttnn.matmul(in0_t, in1_t, program_config=program_config, compute_kernel_config=ckc)
        ttnn.synchronize_device(device)

        # Per-iter measurement: each iter is one matmul, timed individually
        # so we can produce a distribution rather than a single mean.
        per_iter_tflops = []
        for _ in range(MEASURE_ITERS):
            t0 = time.perf_counter()
            ttnn.matmul(in0_t, in1_t, program_config=program_config, compute_kernel_config=ckc)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tf = FLOPS_PER_ITER / elapsed / 1e12 if elapsed > 0 else 0.0
            per_iter_tflops.append(tf)
        return per_iter_tflops
    finally:
        ttnn.close_device(device)


def _summarize(per_iter):
    """Return (median, p10, p90) of per-iter measurements."""
    sorted_ti = sorted(per_iter)
    n = len(sorted_ti)
    if n == 0:
        return (0.0, 0.0, 0.0)
    med = statistics.median(sorted_ti)
    # Use nearest-rank percentile to avoid stats import quirks across pythons.
    p10 = sorted_ti[max(0, int(0.10 * n) - 1)]
    p90 = sorted_ti[min(n - 1, int(0.90 * n))]
    return (med, p10, p90)


def _write_csv(out_dir, rows):
    """Write per-iteration rows to <out_dir>/microbench_phase21c.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "microbench_phase21c.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "iter", "tflops"])
        for case, i, tf in rows:
            w.writerow([case, i, f"{tf:.6f}"])
    return csv_path


def _spawn_case(case):
    """Re-exec self with the right env var to run a single case in a clean process.

    Returns list of per-iter TFLOPs printed by the child. Stdout protocol:
    each measurement line is "TFLOPS <iter> <value>". Anything else is logged.
    """
    env = os.environ.copy()
    if case == "on":
        env["TTNVTOP_REGISTER_PROGRAMS"] = "1"
    else:
        env.pop("TTNVTOP_REGISTER_PROGRAMS", None)
    args = [sys.executable, os.path.abspath(__file__), f"--sampler-{case}", "--_child"]
    proc = subprocess.run(args, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"[microbench] child for case={case} exited {proc.returncode}")
    per_iter = []
    for line in proc.stdout.splitlines():
        if line.startswith("TFLOPS "):
            parts = line.split()
            try:
                per_iter.append(float(parts[2]))
            except (IndexError, ValueError):
                pass
        else:
            print(line)
    return per_iter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sampler-on", action="store_true", help="Run with TTNVTOP_REGISTER_PROGRAMS=1.")
    g.add_argument("--sampler-off", action="store_true", help="Run with TTNVTOP_REGISTER_PROGRAMS unset.")
    g.add_argument(
        "--both",
        action="store_true",
        help="Spawn one subprocess per case so each runs in a clean env, then compare.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to put microbench_phase21c.csv. Default: runs/<UTC-timestamp>/.",
    )
    ap.add_argument(
        "--_child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or Path(f"runs/{ts}")

    if args.sampler_on:
        case = "on"
    elif args.sampler_off:
        case = "off"
    else:
        case = None  # --both

    if args._child or case is not None and not args.both:
        # Single-case run. Either we're a child of --both, or the user invoked
        # --sampler-{on,off} directly. The env var must already be (un)set in
        # this process — we don't manipulate it here, only honor what's set.
        env_state = os.environ.get("TTNVTOP_REGISTER_PROGRAMS", "<unset>")
        print(f"[microbench:{case}] TTNVTOP_REGISTER_PROGRAMS={env_state}")
        per_iter = _run_one_case(case)
        med, p10, p90 = _summarize(per_iter)
        print(f"[microbench:{case}] median {med:.2f} TF   p10 {p10:.2f}   p90 {p90:.2f}   n={len(per_iter)}")
        if args._child:
            # Emit one parseable line per iter so the parent can collect them.
            for i, tf in enumerate(per_iter):
                print(f"TFLOPS {i} {tf:.6f}")
            return 0
        # Direct (non-child) single-case run: write a CSV containing just this case.
        csv_path = _write_csv(out_dir, [(case, i, tf) for i, tf in enumerate(per_iter)])
        print(f"[microbench:{case}] wrote {csv_path}")
        return 0

    # --both: spawn two child processes back to back, compare.
    on_iters = _spawn_case("on")
    off_iters = _spawn_case("off")
    on_med, on_p10, on_p90 = _summarize(on_iters)
    off_med, off_p10, off_p90 = _summarize(off_iters)
    rows = [("on", i, tf) for i, tf in enumerate(on_iters)] + [("off", i, tf) for i, tf in enumerate(off_iters)]
    csv_path = _write_csv(out_dir, rows)

    # Regression frame: how much TF did we lose by enabling the sampler?
    # Positive number => ON is slower than OFF. Use a safe denom.
    if off_med > 0:
        regression_pct = max(0.0, (off_med - on_med) / off_med * 100.0)
    else:
        regression_pct = float("nan")
    verdict = "PASS" if regression_pct < REGRESSION_TARGET_PCT else "FAIL"

    print()
    print(f"ttnvtop sampler ON:   median {on_med:.1f} TF   p10 {on_p10:.1f}   p90 {on_p90:.1f}")
    print(f"ttnvtop sampler OFF:  median {off_med:.1f} TF   p10 {off_p10:.1f}   p90 {off_p90:.1f}")
    print(f"regression: {regression_pct:.2f}%    (target: <{REGRESSION_TARGET_PCT:.1f}%)")
    print(f"verdict: {verdict}")
    print(f"per-iter CSV: {csv_path}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
