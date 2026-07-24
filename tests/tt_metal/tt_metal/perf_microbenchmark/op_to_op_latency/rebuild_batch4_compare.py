#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare batch 4 (grid CB + writer flush-on-pressure) vs batch 1 (dg_v4)."""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = SCRIPT_DIR.parents[4]
BATCH1 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/chart_sweep/dg_v4/chart_data.csv"
BATCH4 = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch4/chart_data.csv"
OUT = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch4/batch4_vs_batch1.csv"


def load_chart(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    rows: dict[int, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            n = int(row["num_cores"])
            rows[n] = row
    return rows


def f(row: dict, key: str, default: str = "") -> str:
    for k in (key, key.replace("_gbs", "_gbps")):
        if k in row and row[k] not in ("", "nan"):
            return row[k]
    return default


def main() -> int:
    b1 = load_chart(BATCH1)
    b4 = load_chart(BATCH4)
    if not b4:
        print(f"No batch 4 data at {BATCH4}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cores",
        "b1_peak_gbs",
        "b4_peak_gbs",
        "peak_delta_pct",
        "b1_cb",
        "b4_cb",
        "b1_op2op_us",
        "b4_op2op_us",
        "op2op_delta_us",
        "b1_dg_median_ns",
        "b4_dg_median_ns",
        "dg_delta_ns",
    ]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cores in sorted(b4.keys()):
            r4 = b4[cores]
            r1 = b1.get(cores, {})
            p1 = float(f(r1, "peak_input_gbps", "0") or 0)
            p4 = float(r4.get("peak_bw_gbs", 0) or 0)
            o1 = float(f(r1, "op2op_us_median", "nan") or "nan")
            o4 = float(r4.get("op2op_us_median", "nan") or "nan")
            d1 = float(f(r1, "dg_median_ns", "nan") or "nan")
            d4 = float(r4.get("dg_median_ns", "nan") or "nan")
            peak_delta = ((p4 - p1) / p1 * 100.0) if p1 > 0 else float("nan")
            w.writerow(
                {
                    "cores": cores,
                    "b1_peak_gbs": f"{p1:.2f}" if p1 else "",
                    "b4_peak_gbs": f"{p4:.2f}" if p4 else "",
                    "peak_delta_pct": f"{peak_delta:+.1f}" if p1 > 0 else "",
                    "b1_cb": r1.get("cb_label", ""),
                    "b4_cb": r4.get("cb_label", ""),
                    "b1_op2op_us": f"{o1:.3f}" if o1 == o1 else "",
                    "b4_op2op_us": f"{o4:.3f}" if o4 == o4 else "",
                    "op2op_delta_us": f"{o4 - o1:+.3f}" if o1 == o1 and o4 == o4 else "",
                    "b1_dg_median_ns": f"{d1:.0f}" if d1 == d1 else "",
                    "b4_dg_median_ns": f"{d4:.0f}" if d4 == d4 else "",
                    "dg_delta_ns": f"{d4 - d1:+.0f}" if d1 == d1 and d4 == d4 else "",
                }
            )

    print(f"Wrote {OUT}")
    with OUT.open() as fh:
        print(fh.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
