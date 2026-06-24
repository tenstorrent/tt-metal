#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decompose the op-to-op gap into its sub-components using existing batch8
profile_log_device_op_to_op_complete.csv files. No new measurements needed —
the device profiler already records every timestamp we need.

Gap timeline (chronological, kernel N -> kernel N+1):

   pack_finish(N)                          <-- gap starts (compute N done)
       |
       |  A. WRITER_TAIL = brisc_done(N) - pack_finish(N)
       |     "How long after compute is done does the writer take to exit?"
       |     This is where the end-of-kernel barrier wait lives, plus any
       |     residual writes left to drain from the CB.
       v
   brisc_done(N)
       |
       |  B. DISPATCH = brisc_go(N+1) - brisc_done(N) = brisc_done_to_go_us (= "dg")
       |     Pure dispatch path. Includes init-register load for kernel N+1.
       |     ** This is what HW init-register virtualization attacks. **
       v
   brisc_go(N+1)
       |
       |  C. READER_FIRST_DRAM = read_after_barrier(N+1) - brisc_go(N+1)
       |     BRISC startup + first DRAM read issued + DRAM round-trip for tile 0.
       v
   read_after_barrier(N+1)
       |
       |  D. DRAM_TO_UNPACK = read_after_to_unpack_tile0_us
       |     First tile in L1 -> compute pulls it.
       v
   unpack_tile0_start(N+1)                 <-- gap ends (compute N+1 starts)

   gap_us = A + B + C + D  (verified within rounding)
"""

from __future__ import annotations

import csv
import glob
import os
import statistics as st
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = SCRIPT_DIR.parents[4]
RUNS_BASE = TT_METAL_HOME / "generated/profiler/op_to_op_runs"
BATCH_PREFIX = os.environ.get("BATCH_PREFIX", "batch8")
OUT_DIR = RUNS_BASE / os.environ.get("OUT_DIR_NAME", BATCH_PREFIX)
OUT_CSV = OUT_DIR / "gap_decomposition.csv"
MIN_PROG_ID = int(os.environ.get("MIN_PROG_ID", "3"))

CORE_COUNTS = [1, 2, 4, 10, 20, 40, 80, 110]
MODES = [(0, "barrier"), (1, "flushed")]


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v) if v not in ("", "nan", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def decompose_row(row: dict) -> dict:
    pack_finish = _f(row, "pack_finish_us")
    brisc_done = _f(row, "brisc_done_us")
    brisc_go = _f(row, "brisc_go_us")
    read_after = _f(row, "read_after_barrier_us")

    A = brisc_done - pack_finish
    B = _f(row, "brisc_done_to_go_us")
    C = read_after - brisc_go
    D = _f(row, "read_after_to_unpack_tile0_us")
    gap = _f(row, "gap_us")

    return {
        "A_writer_tail": A,
        "B_dispatch": B,
        "C_reader_first_dram": C,
        "D_dram_to_unpack": D,
        "gap_us": gap,
    }


def load_runs(cores: int, mode: int) -> list[dict]:
    pattern = str(RUNS_BASE / f"{BATCH_PREFIX}_c{cores}_m{mode}" / "run_*" / "profile_log_device_op_to_op_complete.csv")
    paths = sorted(glob.glob(pattern))
    out: list[dict] = []
    for p in paths:
        with open(p) as fh:
            for row in csv.DictReader(fh):
                try:
                    if int(float(row.get("from_prog_id", 0))) < MIN_PROG_ID:
                        continue
                except (TypeError, ValueError):
                    continue
                out.append(row)
    return out


def median(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return st.median(values) if values else float("nan")


def aggregate(rows: list[dict]) -> dict:
    decomp = [decompose_row(r) for r in rows]
    keys = decomp[0].keys() if decomp else []
    return {k: median([d[k] for d in decomp]) for k in keys}


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if v == v else ""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "cores",
        "mode",
        "gap_us",
        "A_writer_tail_us",
        "A_pct",
        "B_dispatch_us",
        "B_pct",
        "C_reader_first_dram_us",
        "C_pct",
        "D_dram_to_unpack_us",
        "D_pct",
        "n_transitions",
    ]
    table_rows = []

    for cores in CORE_COUNTS:
        for mode_id, mode_label in MODES:
            rows = load_runs(cores, mode_id)
            if not rows:
                continue
            agg = aggregate(rows)
            gap = agg["gap_us"]

            def pct(v):
                return f"{(v / gap) * 100:.1f}" if gap and gap == gap and v == v else ""

            table_rows.append(
                {
                    "cores": cores,
                    "mode": mode_label,
                    "gap_us": fmt(gap),
                    "A_writer_tail_us": fmt(agg["A_writer_tail"]),
                    "A_pct": pct(agg["A_writer_tail"]),
                    "B_dispatch_us": fmt(agg["B_dispatch"]),
                    "B_pct": pct(agg["B_dispatch"]),
                    "C_reader_first_dram_us": fmt(agg["C_reader_first_dram"]),
                    "C_pct": pct(agg["C_reader_first_dram"]),
                    "D_dram_to_unpack_us": fmt(agg["D_dram_to_unpack"]),
                    "D_pct": pct(agg["D_dram_to_unpack"]),
                    "n_transitions": str(len(rows)),
                }
            )

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)

    print(f"\nWrote {OUT_CSV}\n")

    # Print compact summary: just the 4 top-level components per (cores,mode)
    print(
        f"{'cores':>6} {'mode':>8} {'gap us':>8} | {'A writer':>9} ({'%':>4}) {'B disp':>7} ({'%':>4}) {'C rdr+DRAM':>11} ({'%':>4}) {'D rd→cmp':>9} ({'%':>4})"
    )
    print(
        f"{'-'*6:>6} {'-'*8:>8} {'-'*8:>8} | {'-'*9:>9}  {'-'*4:>4}  {'-'*7:>7}  {'-'*4:>4}  {'-'*11:>11}  {'-'*4:>4}  {'-'*9:>9}  {'-'*4:>4}"
    )
    for r in table_rows:
        print(
            f"{r['cores']:>6} {r['mode']:>8} {r['gap_us']:>8} | "
            f"{r['A_writer_tail_us']:>9} ({r['A_pct']:>3}%) "
            f"{r['B_dispatch_us']:>7} ({r['B_pct']:>3}%) "
            f"{r['C_reader_first_dram_us']:>11} ({r['C_pct']:>3}%) "
            f"{r['D_dram_to_unpack_us']:>9} ({r['D_pct']:>3}%)"
        )

    # Per-component delta when switching mode 0 (barrier) -> mode 1 (flushed).
    # Tells us *which* sub-component shrunk to deliver B8's win.
    print(f"\n{'-'*94}")
    print("Where Batch 8's mode-1 win comes from (Δ = mode_1 - mode_0, in us):")
    print(f"{'cores':>6} {'Δ gap':>9} | {'Δ A writer':>11} {'Δ B disp':>10} {'Δ C rdr+DRAM':>13} {'Δ D rd→cmp':>12}")
    for cores in CORE_COUNTS:
        m0 = next((r for r in table_rows if r["cores"] == cores and r["mode"] == "barrier"), None)
        m1 = next((r for r in table_rows if r["cores"] == cores and r["mode"] == "flushed"), None)
        if not m0 or not m1:
            continue

        def d(key):
            a = float(m1[key] or "nan")
            b = float(m0[key] or "nan")
            return a - b if a == a and b == b else float("nan")

        print(
            f"{cores:>6} {d('gap_us'):>+9.3f} | "
            f"{d('A_writer_tail_us'):>+11.3f} "
            f"{d('B_dispatch_us'):>+10.3f} "
            f"{d('C_reader_first_dram_us'):>+13.3f} "
            f"{d('D_dram_to_unpack_us'):>+12.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
