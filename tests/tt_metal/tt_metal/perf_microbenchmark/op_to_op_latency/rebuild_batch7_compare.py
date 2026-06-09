#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare batch 7 (grid CB + flush-on-pressure + cross-program DRAM offset)
against batch 4 (same, but every program touches the same DRAM slice) and
batch 1 (sequential CB tune baseline).

The batch7-vs-batch4 delta isolates the impact of forcing the DRAM controller
to open new rows per program enqueue (i.e. how much of batch 4's apparent
op-to-op gap was being optimistically helped by warm-row reuse)."""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = SCRIPT_DIR.parents[4]

BATCH1 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/chart_sweep/dg_v4/chart_data.csv"
BATCH4 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch4/chart_data.csv"
BATCH7 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch7/chart_data.csv"
OUT = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch7/batch7_vs_batch4.csv"


def load(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    out: dict[int, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["num_cores"])] = row
    return out


def gbs(row: dict) -> float:
    for k in ("peak_bw_gbs", "peak_input_gbps"):
        if k in row and row[k]:
            return float(row[k])
    return 0.0


def gap(row: dict) -> float:
    return float(row.get("op2op_us_median", "nan") or "nan")


def dg(row: dict) -> float:
    return float(row.get("dg_median_ns", "nan") or "nan")


def cb(row: dict) -> str:
    return row.get("cb_label", "")


def main() -> int:
    b1, b4, b7 = load(BATCH1), load(BATCH4), load(BATCH7)
    if not b7:
        print(f"No batch 7 data at {BATCH7}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cores",
        "b1_peak_gbs",
        "b4_peak_gbs",
        "b7_peak_gbs",
        "b7_vs_b4_bw_pct",
        "b1_cb",
        "b4_cb",
        "b7_cb",
        "b1_op2op_us",
        "b4_op2op_us",
        "b7_op2op_us",
        "b7_vs_b4_gap_us",
        "b7_vs_b4_gap_pct",
        "b1_dg_ns",
        "b4_dg_ns",
        "b7_dg_ns",
        "b7_vs_b4_dg_ns",
    ]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cores in sorted(b7.keys()):
            r1, r4, r7 = b1.get(cores, {}), b4.get(cores, {}), b7[cores]
            p1, p4, p7 = gbs(r1), gbs(r4), gbs(r7)
            o1, o4, o7 = gap(r1), gap(r4), gap(r7)
            d1, d4, d7 = dg(r1), dg(r4), dg(r7)
            w.writerow(
                {
                    "cores": cores,
                    "b1_peak_gbs": f"{p1:.2f}" if p1 else "",
                    "b4_peak_gbs": f"{p4:.2f}" if p4 else "",
                    "b7_peak_gbs": f"{p7:.2f}" if p7 else "",
                    "b7_vs_b4_bw_pct": f"{(p7 - p4) / p4 * 100:+.1f}" if p4 > 0 and p7 else "",
                    "b1_cb": cb(r1),
                    "b4_cb": cb(r4),
                    "b7_cb": cb(r7),
                    "b1_op2op_us": f"{o1:.3f}" if o1 == o1 else "",
                    "b4_op2op_us": f"{o4:.3f}" if o4 == o4 else "",
                    "b7_op2op_us": f"{o7:.3f}" if o7 == o7 else "",
                    "b7_vs_b4_gap_us": f"{o7 - o4:+.3f}" if o4 == o4 and o7 == o7 else "",
                    "b7_vs_b4_gap_pct": (f"{(o7 - o4) / o4 * 100:+.1f}" if o4 == o4 and o7 == o7 and o4 != 0 else ""),
                    "b1_dg_ns": f"{d1:.0f}" if d1 == d1 else "",
                    "b4_dg_ns": f"{d4:.0f}" if d4 == d4 else "",
                    "b7_dg_ns": f"{d7:.0f}" if d7 == d7 else "",
                    "b7_vs_b4_dg_ns": f"{d7 - d4:+.0f}" if d4 == d4 and d7 == d7 else "",
                }
            )

    print(f"Wrote {OUT}\n")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
