# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score generated/qkv_split_manifest.json against the newest profiler CSV."""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

MANIFEST = "generated/qkv_split_manifest.json"


def main() -> int:
    man = json.load(open(MANIFEST))
    arms, reps = man["arms"], man["reps"]
    csvs = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not csvs:
        print("no profiler CSV — run under `python -m tracy -p`")
        return 1
    path = csvs[-1]
    if os.path.getmtime(path) < os.path.getmtime(MANIFEST) - 5:
        print(f"REFUSING: newest CSV {path} predates the manifest; the profiler wrote nothing.")
        return 1
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "nlpcreate" in (r.get("OP CODE") or "").lower().replace("_", ""):
                d = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
                if d.isdigit():
                    rows.append((int(d) / 1000.0, int(float(r.get("CORE COUNT") or 0))))
    need = sum(1 + a["reps"] for a in arms)
    print(f"csv: {path}\nnlp_create_qkv_heads rows: {len(rows)}, manifest expects {need}")
    if len(rows) < need:
        print("REFUSING: cannot attribute rows to arms safely.")
        return 1
    rows = rows[-need:]
    i = 0
    print(f"\n| m | config | median us | cores |\n|---:|---|---:|---:|")
    for a in arms:
        chunk = rows[i : i + 1 + a["reps"]]
        i += 1 + a["reps"]
        us = statistics.median(t for t, _ in chunk[1:])
        cores = sorted({c for _, c in chunk[1:]})
        print(f"| {a['m']} | {a['label']} | {us:.1f} | {cores} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
