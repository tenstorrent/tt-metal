# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize tt-perf-report CSVs into doc/full_model/perf_report_summary.json.

Normalizes per *step* using the number of times a once-per-step anchor op
appears in the window, not an assumed iteration count: if the device profiler
dropped markers the anchor count drops with them, so the per-step figures stay
honest and the recorded ``anchor_calls`` shows how many steps were captured.

    python models/autoports/zai_org_glm_4_7_flash/tests/summarize_perf_report.py \\
        --tracy-dir models/autoports/zai_org_glm_4_7_flash/doc/full_model/tracy
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

#: One LM-head matmul runs exactly once per decode step and once per prefill.
ANCHOR = "MatmulDeviceOperation 32 x 2048 x 154880"


def _num(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def summarize(path: Path, anchor: str = ANCHOR):
    rows = list(csv.DictReader(path.open()))
    steps = sum(1 for r in rows if r.get("OP Code", "").strip() == anchor)
    if steps == 0:
        steps = 1
    per_op_time = collections.Counter()
    per_op_count = collections.Counter()
    total = gap = 0.0
    for row in rows:
        op = row.get("OP Code", "").strip()
        dur = _num(row.get("Device Time"))
        per_op_time[op] += dur
        per_op_count[op] += 1
        total += dur
        gap += _num(row.get("Op-to-Op Gap"))
    return {
        "csv": path.name,
        "anchor_op": anchor,
        "anchor_calls_in_window": steps,
        "normalized_by": "anchor_calls_in_window (one LM head per step), not an assumed iteration count",
        "total_rows": len(rows),
        "ops_per_step": round(len(rows) / steps, 1),
        "device_us_per_step": round(total / steps, 1),
        "op_to_op_gap_us_per_step": round(gap / steps, 1),
        "top_ops": [
            {
                "op": op,
                "us_per_step": round(t / steps, 2),
                "pct": round(t / total * 100, 1) if total else 0.0,
                "calls_per_step": round(per_op_count[op] / steps, 2),
            }
            for op, t in per_op_time.most_common(14)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracy-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.tracy_dir.parent / "perf_report_summary.json")
    summary = {}
    for name in ("decode_model", "decode_tokenout", "prefill"):
        path = args.tracy_dir / f"{name}_perf_report.csv"
        if not path.is_file():
            continue
        summary[name] = summarize(path)
        s = summary[name]
        print(
            f"== {name}: {s['anchor_calls_in_window']} captured steps, "
            f"{s['ops_per_step']} ops/step, {s['device_us_per_step']} us/step device, "
            f"{s['op_to_op_gap_us_per_step']} us/step gap"
        )
        for row in s["top_ops"][:8]:
            print(f"   {row['op']:46s} {row['us_per_step']:9.1f} us/step {row['pct']:5.1f}%  x{row['calls_per_step']}")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
