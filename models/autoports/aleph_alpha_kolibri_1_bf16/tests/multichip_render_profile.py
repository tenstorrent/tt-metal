# SPDX-License-Identifier: Apache-2.0
"""Extract signposted rows by file order and render each device independently.

Trace replay rows retain capture host timestamps: timestamp filtering would
silently exclude them. Multi-device merging also obscures operation gaps.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def run(directory, preserve_execution_order=False, prefill_active_experts=None):
    source = max((directory / "reports").glob("*/ops_perf_results_*.csv"))
    with source.open() as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    provenance = dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection="strict CSV row interval between signposts, then DEVICE ID; no host timestamp filter",
        runs=[],
    )
    for mode in ("prefill", "decode"):
        first = next(i for i, r in enumerate(rows) if r["OP CODE"] == "PERF_" + mode.upper())
        last = next(i for i, r in enumerate(rows) if r["OP CODE"] == "PERF_" + mode.upper() + "_END")
        for device in ("0", "1", "2", "3"):
            selected = [r for r in rows[first + 1 : last] if r["DEVICE ID"] == device]
            assert selected, (mode, device)
            if mode == "decode":
                assert all(r["METAL TRACE REPLAY SESSION ID"] for r in selected)
            target = directory / f"{mode}_device{device}_ops.csv"
            with target.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(selected)
            base = ["tt-perf-report", str(target)]
            if preserve_execution_order:
                # Mixed eager-prefill + traced terminal intervals must retain
                # CSV execution order: replay keeps capture host timestamps.
                base += ["--tracing-mode"]
            if mode == "decode":
                base += ["--active-experts", "6"]
            elif prefill_active_experts is not None:
                base += ["--active-experts", str(prefill_active_experts)]
            report = directory / f"{mode}_device{device}_report"
            commands = []
            for suffix, args in [(".txt", ["--no-color", "--no-summary"]), (".log", ["--csv", str(report) + ".csv"])]:
                command = base + args
                commands.append(command)
                with open(str(report) + suffix, "w") as f:
                    subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
            provenance["runs"].append(
                dict(
                    mode=mode,
                    device=device,
                    rows=len(selected),
                    commands=commands,
                    kernel_us=sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in selected) / 1000,
                    first_row=first,
                    last_row=last,
                )
            )
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--tracing-mode", action="store_true")
    p.add_argument("--prefill-active-experts", type=int)
    a = p.parse_args()
    run(a.directory, preserve_execution_order=a.tracing_mode, prefill_active_experts=a.prefill_active_experts)
