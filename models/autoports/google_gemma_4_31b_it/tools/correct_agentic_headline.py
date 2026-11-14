#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate original headline, then collect only its matching96-request correction."""

import argparse
import subprocess
import sys
from pathlib import Path

from remeasure_agentic_rates import read, report, save, validate_raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement", type=Path, required=True)
    p.add_argument("--correction", type=Path, required=True)
    p.add_argument("--original-rates", type=Path, required=True)
    p.add_argument("--server-identity", type=Path, required=True)
    p.add_argument("--expected-precision", type=Path, required=True)
    p.add_argument("--stop-file", type=Path, required=True)
    p.add_argument("--budget-seconds", type=float, default=1200)
    args = p.parse_args()
    original = read(args.original_rates)
    canonical = read(args.measurement / "serving_rates.json")
    if read(args.measurement / "run.json")["status"] != "completed" or len(canonical["rows"]) != 15:
        raise ValueError("Active15-row run must finish successfully before correction")
    row_id = "b32-c32-i4096-gpqa"
    row = next(r for r in original["rows"] if r["id"] == row_id)
    rawpath = args.original_rates.parent / row["raw"]
    old = read(rawpath)
    validate_raw(old, row, 96)
    warm = read(rawpath.with_name("perf-b32-warmup.json"))
    validate_raw(warm, row, 32)
    seeds = []
    for name in ["perf-b32.json.command.json", "perf-b32-warmup.json.command.json"]:
        argv = read(rawpath.with_name(name))["argv"]
        seeds.append(int(argv[argv.index("--seed") + 1]))
    if seeds != [4132, 4133]:
        raise ValueError("Unexpected original headline dataset seeds")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("remeasure_agentic_rates.py")),
            "--output",
            str(args.correction),
            "--expected-precision",
            str(args.expected_precision),
            "--server-identity",
            str(args.server_identity),
            "--original-rates",
            str(args.original_rates),
            "--budget-seconds",
            str(args.budget_seconds),
            "--row-timeout-seconds",
            "600",
            "--stop-file",
            str(args.stop_file),
            "--row-id",
            row_id,
            "--dataset-seed",
            "4132",
        ],
        check=True,
    )
    corrected = read(args.correction / "serving_rates.json")
    if corrected["status"] != "completed" or len(corrected["rows"]) != 1:
        raise ValueError("Correction incomplete")
    new = corrected["rows"][0]
    if new["requests"] != 96 or new["completed"] != 96:
        raise ValueError("Wrong correction population")
    if canonical["precision_sha256"] != corrected["precision_sha256"]:
        raise ValueError("Candidate policy changed")
    # Preserve the complete pre-correction reports, including the confounded32 row.
    archive = args.measurement / "before-headline-correction"
    archive.mkdir(exist_ok=False)
    for name in [
        "serving_rates.json",
        "serving_rates-gpqa.json",
        "serving_rates-swe.json",
        "serving_rates-by-config.json",
    ]:
        (archive / name).write_bytes((args.measurement / name).read_bytes())
    new[
        "correction_note"
    ] = "96 measured/32warmup, original seeds4132/4133; supersedes confounded32-request canonical row, whose raw evidence and previous reports are preserved."
    rows = [new if r["id"] == row_id else r for r in canonical["rows"]]
    report(
        args.measurement,
        original,
        rows,
        corrected["server_config"],
        read(args.expected_precision),
        [r["id"] for r in rows],
    )
    save(
        args.measurement / "headline-correction.json",
        {
            "row": row_id,
            "original_raw": str(rawpath),
            "correction": str(args.correction),
            "preserved_reports": str(archive),
            "original_measured_requests": 96,
            "warmup_requests": 32,
            "dataset_seeds": seeds,
        },
    )


if __name__ == "__main__":
    main()
