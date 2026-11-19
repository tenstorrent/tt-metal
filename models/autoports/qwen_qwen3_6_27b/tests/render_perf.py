# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only rendering of completed Tracy runs and explicit-unit timing totals."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advice", action="store_true")
    parser.add_argument(
        "--plain-artifacts", action="store_true", help="Keep CSV/logs without per-artifact hashing or compression"
    )
    parser.add_argument("--tracy-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = sorted(args.tracy_root.glob("reports/*/ops_perf_results_*.csv"))[-1]
    args.output.mkdir(parents=True, exist_ok=True)
    copied = args.output / "ops.csv"
    shutil.copyfile(source, copied)
    raw = copied.read_bytes()
    if not args.plain_artifacts:
        with gzip.open(args.output / "ops.csv.gz", "wb") as stream:
            stream.write(raw)
    measured = json.loads(args.result.read_text())
    result = {
        "provenance": measured["provenance"],
        "correctness_result": str(args.result),
        "raw_ops_source": str(source),
        "raw_ops_sha256": None if args.plain_artifacts else hashlib.sha256(raw).hexdigest(),
        "tool": "tt-perf-report " + importlib.metadata.version("tt-perf-report"),
        "architecture_argument": "blackhole (P300c; tool models 1.35 GHz and 512 GB/s per chip)",
        "batch": measured["batch"],
        "prefill_length": measured["length"],
        "decode_context": measured["length"] + 1,
        "commands": [],
    }
    for mode in ("prefill", "decode"):
        stem = args.output / (mode + "_perf_report")
        command = [
            "tt-perf-report",
            str(copied),
            "--start-signpost",
            "PERF_" + mode.upper(),
            "--end-signpost",
            "PERF_" + mode.upper() + "_END",
            "--no-summary",
            "--no-color",
            "--arch",
            "blackhole",
        ]
        if not args.advice and "--no-advice" not in command:
            command.append("--no-advice")
        result["commands"].append(command)
        with stem.with_suffix(".txt").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        command = command + ["--csv", str(stem.with_suffix(".csv"))]
        result["commands"].append(command)
        with stem.with_suffix(".console.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        with stem.with_suffix(".csv").open() as stream:
            rows = [r for r in csv.DictReader(stream) if r["ID"].isdigit() and r["Device Time [μs]"]]
        kernel = sum(float(r["Device Time [μs]"]) for r in rows)
        gap = sum(float(r["Op-to-Op Gap [μs]"] or 0) for r in rows)
        overlap = sum(float(r.get("Overlap [μs]") or 0) for r in rows)
        result[mode] = {
            "device_kernel_sum_us": kernel,
            "op_gap_sum_us": gap,
            "overlap_sum_us": overlap,
            "kernel_plus_gap_us": kernel + gap,
            "device_ops": len(rows),
            "unit": "microseconds",
            "device_time_column": "Device Time [μs]",
        }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({mode: result[mode] for mode in ("prefill", "decode")}, indent=2))


if __name__ == "__main__":
    main()
