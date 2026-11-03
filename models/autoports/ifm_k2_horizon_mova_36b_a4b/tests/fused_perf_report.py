# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve raw profiler input, rendered tables, filtered CSVs and provenance."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .run_decoder import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("layer", type=int)
ap.add_argument("ops_csv", type=Path)
ap.add_argument("--tag", default="final")
args = ap.parse_args()
kind = "dense" if args.layer == 0 else "mova_moe"
dest = ROOT / "doc/fused_decoder/tracy" / args.tag / kind
dest.mkdir(parents=True, exist_ok=True)
provenance = dict(
    layer=args.layer,
    source_csv=str(args.ops_csv.resolve()),
    source_sha256=hashlib.sha256(args.ops_csv.read_bytes()).hexdigest(),
    windows={},
)
for mode in ("prefill", "decode"):
    raw = dest / f"{mode}_ops.csv"
    shutil.copy2(args.ops_csv, raw)
    base = [
        "tt-perf-report",
        str(raw),
        "--start-signpost",
        f"PERF_{mode.upper()}",
        "--end-signpost",
        f"PERF_{mode.upper()}_END",
        "--no-advice",
        "--no-color",
    ]
    filtered = dest / f"{mode}_perf_report.csv"
    csv_command = base + ["--csv", str(filtered)]
    with (dest / f"{mode}_perf_report.console.log").open("w") as f:
        subprocess.run(csv_command, stdout=f, stderr=subprocess.STDOUT, check=True)
    table_command = base + ["--no-summary"]
    with (dest / f"{mode}_perf_report.txt").open("w") as f:
        subprocess.run(table_command, stdout=f, stderr=subprocess.STDOUT, check=True)
    with filtered.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, f"Empty {mode} profiling window"
    time_column = "Device Time [μs]"
    assert time_column in rows[0], "Review report units before calculating latency"
    provenance["windows"][mode] = dict(
        csv_command=csv_command,
        table_command=table_command,
        csv_columns=list(rows[0]),
        rows=len(rows),
        device_time_column=time_column,
        device_time_unit="microseconds",
        device_kernel_sum_ms=sum(float(row[time_column] or 0) for row in rows) / 1000,
        op_gap_sum_ms=sum(float(row["Op-to-Op Gap [μs]"] or 0) for row in rows) / 1000,
    )
(dest / "provenance.json").write_text(json.dumps(provenance, indent=2))
print(json.dumps(provenance, indent=2))
