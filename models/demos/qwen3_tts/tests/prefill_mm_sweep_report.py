# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score generated/prefill_mm_manifest.json against the newest profiler CSV."""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

MANIFEST = "generated/prefill_mm_manifest.json"
PEAK = 288.0
BFP8 = 1.0625


def main() -> int:
    man = json.load(open(MANIFEST))
    arms, reps = man["arms"], man["reps"]
    csvs = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not csvs:
        print("no CSV — run under `python -m tracy -p`")
        return 1
    path = csvs[-1]
    if os.path.getmtime(path) < os.path.getmtime(MANIFEST) - 5:
        print(f"REFUSING: {path} predates the manifest.")
        return 1
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "matmul" in (r.get("OP CODE") or "").lower():
                d = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
                if d.isdigit():
                    rows.append(int(d) / 1000.0)
    need = sum(1 + a["reps"] for a in arms)
    print(f"csv: {path}\nmatmul rows {len(rows)}, manifest expects {need}")
    if len(rows) < need:
        print("REFUSING: cannot attribute.")
        return 1
    rows = rows[-need:]
    out, i = [], 0
    for a in arms:
        chunk = rows[i : i + 1 + a["reps"]]
        i += 1 + a["reps"]
        a["us"] = statistics.median(chunk[1:])
        out.append(a)
    by = {}
    for a in out:
        by.setdefault((a["shape"], a["m"]), []).append(a)
    for (shape, m), lst in by.items():
        K, N = lst[0]["K"], lst[0]["N"]
        wb = K * N * BFP8
        base = next((a for a in lst if a["shipped"]), None)
        best = min(lst, key=lambda a: a["us"])
        print(f"\n### {shape}  M={m} K={K} N={N}   weight {wb/1e6:.2f} MB")
        print("| config | us | GB/s | % peak |\n|---|---:|---:|---:|")
        for a in sorted(lst, key=lambda a: a["us"]):
            g = wb / (a["us"] * 1e-6) / 1e9
            tag = "  <-- SHIPPED" if a["shipped"] else ""
            print(f"| {a['label']} | {a['us']:.1f} | {g:.0f} | {100*g/PEAK:.1f}{tag} |")
        if base and best is not base:
            d = base["us"] - best["us"]
            print(f"  -> best beats shipped by {d:.1f} us/matmul")
    return 0


if __name__ == "__main__":
    sys.exit(main())
