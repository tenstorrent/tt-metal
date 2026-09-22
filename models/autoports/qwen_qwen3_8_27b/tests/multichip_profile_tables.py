# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Retain per-device rows to avoid cross-device timestamp gaps in merged reports."""

import argparse
import csv
import gzip
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
source = max(a.directory.glob("reports/*/ops_perf_results*.csv*"), key=lambda path: path.parent.name)
with gzip.open(source, "rt") if source.suffix == ".gz" else source.open() as f:
    reader = csv.DictReader(f)
    names, rows = reader.fieldnames, list(reader)
for device in sorted({int(r["DEVICE ID"]) for r in rows if r["DEVICE ID"]}):
    path = a.directory / f"device{device}_ops.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(r for r in rows if r["DEVICE ID"] in ("", str(device)))
    for phase in ("prefill", "decode"):
        base = a.directory / f"device{device}_{phase}_perf_report"
        command = [
            "python_env/bin/tt-perf-report",
            str(path),
            "--start-signpost",
            f"PERF_{phase.upper()}",
            "--end-signpost",
            f"PERF_{phase.upper()}_END",
        ]
        with base.with_suffix(".txt").open("w") as out:
            subprocess.run(command + ["--no-summary"], stdout=out, stderr=subprocess.STDOUT, check=True)
        with base.with_suffix(".console.log").open("w") as out:
            subprocess.run(
                command + ["--csv", str(base.with_suffix(".csv"))], stdout=out, stderr=subprocess.STDOUT, check=True
            )
