# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
"""
Export DeepSeek fused-op device perf artifacts to an Excel-friendly CSV.

This script reads benchmark artifact files (partial_run_*.pkl) and writes a
flat CSV with one row per fused-op step, including:
  - step metadata (mode, seq_len)
  - performance metrics (kernel/op-to-op/total in us)
  - run metadata (timestamps, commit, device info)

Usage:
  python3 models/demos/deepseek_v3/tests/fused_op_unit_tests/export_device_perf_csv.py
  python3 .../export_device_perf_csv.py --input-glob "generated/benchmark_data/partial_run_*.pkl" \
      --output "generated/benchmark_data/deepseek_device_perf_summary.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import pickle
import re
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"(?P<mode>decode|prefill)_seq(?P<seq>\d+)")


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_step_fields(step_name: str) -> tuple[str, str]:
    match = STEP_RE.search(step_name or "")
    if not match:
        return "", ""
    return match.group("mode"), match.group("seq")


def _collect_rows(input_glob: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found for glob: {input_glob}")

    for path in files:
        with open(path, "rb") as fp:
            run = pickle.load(fp)

        per_step: dict[str, dict[str, Any]] = {}
        measurements = getattr(run, "measurements", [])
        for m in measurements:
            step_name = _to_str(getattr(m, "step_name", ""))
            if not step_name:
                continue

            entry = per_step.setdefault(
                step_name,
                {
                    "kernel_us": None,
                    "op_to_op_us": None,
                },
            )

            metric_name = _to_str(getattr(m, "name", ""))
            metric_value = getattr(m, "value", None)
            if metric_name == "total_kernel_duration_us":
                entry["kernel_us"] = float(metric_value)
            elif metric_name == "total_op_to_op_latency_us":
                entry["op_to_op_us"] = float(metric_value)

        for step_name, metrics in per_step.items():
            mode, seq_len = _extract_step_fields(step_name)
            kernel_us = metrics["kernel_us"]
            op_to_op_us = metrics["op_to_op_us"]
            total_us = ""
            if kernel_us is not None and op_to_op_us is not None:
                total_us = kernel_us + op_to_op_us

            rows.append(
                {
                    "artifact_file": path,
                    "step_name": step_name,
                    "mode": mode,
                    "seq_len": seq_len,
                    "kernel_us": "" if kernel_us is None else kernel_us,
                    "op_to_op_us": "" if op_to_op_us is None else op_to_op_us,
                    "total_us": total_us,
                    "run_start_ts": _to_str(getattr(run, "run_start_ts", "")),
                    "run_end_ts": _to_str(getattr(run, "run_end_ts", "")),
                    "git_commit_hash": _to_str(getattr(run, "git_commit_hash", "")),
                    "git_branch_name": _to_str(getattr(run, "git_branch_name", "")),
                    "device_info": _to_str(getattr(run, "device_info", "")),
                    "device_hostname": _to_str(getattr(run, "device_hostname", "")),
                }
            )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export DeepSeek device perf artifacts to CSV")
    parser.add_argument(
        "--input-glob",
        default="generated/benchmark_data/partial_run_*.pkl",
        help="Glob pattern for benchmark artifact files",
    )
    parser.add_argument(
        "--output",
        default="generated/benchmark_data/deepseek_device_perf_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = _collect_rows(args.input_glob)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "artifact_file",
        "step_name",
        "mode",
        "seq_len",
        "kernel_us",
        "op_to_op_us",
        "total_us",
        "run_start_ts",
        "run_end_ts",
        "git_commit_hash",
        "git_branch_name",
        "device_info",
        "device_hostname",
    ]

    # utf-8-sig makes Excel reliably detect UTF-8 encoding.
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
