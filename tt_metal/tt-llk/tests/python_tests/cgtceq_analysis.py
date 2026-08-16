# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Post-run analysis for the (Cgt, Ceq) engine bench (test_cgtceq_perf.py).

Standalone, host-only. Reads the RAW combined CSV (not the .post.csv — the
slopes below must divide by nothing) plus the bisect per-row dump, and prints:

  1. Additivity table: per stream arm, cyc/vec under MATH_ISOLATE and
     L1_TO_L1 from a two-point slope over tile_cnt {16, 64} (1 tile = 32
     vectors), plus the Gate-2 delta check
         L1_TO_L1(arm) - L1_TO_L1(none)  vs  MATH_ISOLATE(arm).
  2. Rendezvous 3x3 matrix: cycles/decision =
         (slope(rendezvous[fold,sync]) - slope(rate)) * 64
     with the slope taken over ITER_COUNT {512, 2048} (slope is per vector,
     so the delta is per vector and x64 restores per segment/decision).
  3. Bisection decisions/cycles p50/p95 per distribution from the row dump.

Usage (from tt_metal/tt-llk/tests):
    python python_tests/cgtceq_analysis.py \
        [--csv perf_data/test_cgtceq_perf/test_cgtceq_perf.csv] \
        [--rows /tmp/cgtceq_bisect_rows.txt]

HONESTY GUARD: these constants price the Gate-2 correctness ORACLE only;
they are inputs to RADIX_BUCKET_GPU.md correction #8, not a claimed win.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

VECTORS_PER_TILE = 32
VECTORS_PER_SEGMENT = 64
ITER_POINTS = (512, 2048)
TILE_POINTS = (16, 64)


def _f(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def load_rows(csv_path):
    with open(csv_path) as f:
        return [r for r in csv.DictReader(f)]


def tile_loop_mean(rows, run_col, **match):
    """Mean of `mean(<run_col>)` over TILE_LOOP rows matching the filters."""
    vals = []
    for r in rows:
        if r.get("marker") != "TILE_LOOP":
            continue
        if any(str(r.get(k, "")) != str(v) for k, v in match.items()):
            continue
        v = _f(r, f"mean({run_col})")
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def two_point_slope(rows, run_col, axis_col, points, **match):
    lo = tile_loop_mean(rows, run_col, **{axis_col: points[0]}, **match)
    hi = tile_loop_mean(rows, run_col, **{axis_col: points[1]}, **match)
    if lo is None or hi is None:
        return None
    return (hi - lo) / (points[1] - points[0])


def _fmt(v):
    return "  n/a" if v is None else f"{v:.3f}"


def additivity(rows):
    print("\n=== (i) additivity: cyc per 32-elem vector (slope over tile_cnt) ===")
    arms = ["stream_none", "ctrl_load", "ctrl_swap", "stream_single", "stream_dual"]
    per = {}
    for arm in arms:
        line = {"arm": arm}
        for rt in ("MATH_ISOLATE", "L1_TO_L1", "UNPACK_ISOLATE"):
            s = two_point_slope(rows, rt, "tile_cnt", TILE_POINTS, arm=arm)
            line[rt] = None if s is None else s / VECTORS_PER_TILE
        per[arm] = line
        print(
            f"  {arm:14s} MATH_ISOLATE={_fmt(line['MATH_ISOLATE'])} "
            f"L1_TO_L1={_fmt(line['L1_TO_L1'])} "
            f"UNPACK_ISOLATE={_fmt(line['UNPACK_ISOLATE'])}"
        )
    print("\n  Controls: ctrl_load must be ~1.0 MATH_ISOLATE, ctrl_swap ~2.0 —")
    print("  if not, the feed path is limiting and nothing else is readable.")
    base = per["stream_none"].get("L1_TO_L1")
    for arm in ("stream_single", "stream_dual"):
        l1 = per[arm].get("L1_TO_L1")
        iso = per[arm].get("MATH_ISOLATE")
        if None not in (base, l1, iso):
            print(
                f"  additivity[{arm}]: L1_TO_L1 - floor = {l1 - base:.3f} "
                f"vs MATH_ISOLATE = {iso:.3f} "
                f"({'ADDITIVE' if abs((l1 - base) - iso) < 0.3 else 'NOT additive'})"
            )


def rendezvous(rows):
    print("\n=== (ii) rendezvous: cycles per data-dependent decision ===")
    rate = two_point_slope(rows, "MATH_ISOLATE", "iters", ITER_POINTS, arm="rate")
    if rate is None:
        print("  (no 'rate' rows found)")
        return
    print(f"  rate arm slope: {rate:.4f} cyc/vec (expect ~2.0)")
    print("  fold\\sync      S0=tensix_sync  S1=sem+pcbuf  S2=sentinel")
    for fold, label in (
        (0, "R0 full(read 1)"),
        (1, "R1 part(read16)"),
        (2, "R2 none(read64)"),
    ):
        cells = []
        for sync in (0, 1, 2):
            s = two_point_slope(
                rows,
                "MATH_ISOLATE",
                "iters",
                ITER_POINTS,
                arm="rendezvous",
                fold=fold,
                sync=sync,
            )
            cells.append(
                "   n/a  " if s is None else f"{(s - rate) * VECTORS_PER_SEGMENT:8.1f}"
            )
        print(f"  {label:15s} " + "  ".join(cells))
    print("  (S0/R0 should reproduce the >=25.1-cyc PassSync floor plus the")
    print("   fold+read it adds; prior for the full menu: 25-100 cyc.)")


def bisect_stats(rows_path):
    print("\n=== (iii) bisection p50/p95 per distribution ===")
    p = Path(rows_path)
    if not p.exists():
        print(f"  (no row dump at {p})")
        return
    dec = defaultdict(list)
    cyc = defaultdict(list)
    with open(p) as f:
        for line in f:
            try:
                dist, group, k, sync, r, decisions, cycles, mode = line.strip().split(
                    ","
                )
            except ValueError:
                continue
            key = f"{dist}(k={k},s={sync})"
            dec[key].append(int(decisions))
            cyc[key].append(int(cycles))

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    for key in sorted(dec):
        d, c = dec[key], cyc[key]
        print(
            f"  {key:28s} rows={len(d):3d} decisions p50={pct(d, 0.5):2d} "
            f"p95={pct(d, 0.95):2d}  cycles p50={pct(c, 0.5):6d} p95={pct(c, 0.95):6d}"
        )
    print("  model check: cycles/row ~ decisions x (row count-pass + rendezvous);")
    print("  1 row = 1 tile = 32 vectors, count pass ~64-70 cyc + rendezvous.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="perf_data/test_cgtceq_perf/test_cgtceq_perf.csv",
        help="RAW combined csv (not .post.csv)",
    )
    ap.add_argument("--rows", default="/tmp/cgtceq_bisect_rows.txt")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    additivity(rows)
    rendezvous(rows)
    bisect_stats(args.rows)


if __name__ == "__main__":
    main()
