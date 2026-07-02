#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare batch 5 (grid peak + shrink CB) vs batch 1 and batch 4."""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = SCRIPT_DIR.parents[4]

BATCH1 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/chart_sweep/dg_v4/chart_data.csv"
BATCH4 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch4/chart_data.csv"
BATCH5 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch5/chart_data.csv"
OUT = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch5/batch5_vs_batch1_batch4.csv"


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
    return float(row.get("op2op_us_median", "nan"))


def dg(row: dict) -> float:
    return float(row.get("dg_median_ns", "nan"))


def cb(row: dict) -> str:
    return row.get("cb_label", "")


def main() -> int:
    b1, b4, b5 = load(BATCH1), load(BATCH4), load(BATCH5)
    if not b5:
        print(f"No batch 5 data at {BATCH5}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cores",
        "b1_peak_gbs",
        "b4_peak_gbs",
        "b5_peak_gbs",
        "b5_vs_b1_bw_pct",
        "b1_cb",
        "b4_cb",
        "b5_cb",
        "b1_op2op_us",
        "b4_op2op_us",
        "b5_op2op_us",
        "b5_vs_b1_gap_us",
        "b1_dg_ns",
        "b5_dg_ns",
    ]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cores in sorted(b5.keys()):
            r1, r4, r5 = b1.get(cores, {}), b4.get(cores, {}), b5[cores]
            p1, p4, p5 = gbs(r1), gbs(r4), gbs(r5)
            o1, o4, o5 = gap(r1), gap(r4), gap(r5)
            d1, d5 = dg(r1), dg(r5)
            w.writerow(
                {
                    "cores": cores,
                    "b1_peak_gbs": f"{p1:.2f}" if p1 else "",
                    "b4_peak_gbs": f"{p4:.2f}" if p4 else "",
                    "b5_peak_gbs": f"{p5:.2f}" if p5 else "",
                    "b5_vs_b1_bw_pct": f"{(p5 - p1) / p1 * 100:+.1f}" if p1 > 0 and p5 else "",
                    "b1_cb": cb(r1),
                    "b4_cb": cb(r4),
                    "b5_cb": cb(r5),
                    "b1_op2op_us": f"{o1:.3f}" if o1 == o1 else "",
                    "b4_op2op_us": f"{o4:.3f}" if o4 == o4 else "",
                    "b5_op2op_us": f"{o5:.3f}" if o5 == o5 else "",
                    "b5_vs_b1_gap_us": f"{o5 - o1:+.3f}" if o1 == o1 and o5 == o5 else "",
                    "b1_dg_ns": f"{d1:.0f}" if d1 == d1 else "",
                    "b5_dg_ns": f"{d5:.0f}" if d5 == d5 else "",
                }
            )

    print(f"Wrote {OUT}\n")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
