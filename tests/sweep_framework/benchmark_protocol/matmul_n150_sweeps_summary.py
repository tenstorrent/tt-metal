#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = idx - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def _extract_e2e_ms(row: dict) -> float | None:
    for metric in row.get("metrics", []):
        if metric.get("metric_name") == "e2e_perf_ms":
            try:
                return float(metric.get("metric_value"))
            except (TypeError, ValueError):
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model_traced matmul sweeps_runner and emit compact summary JSON")
    parser.add_argument("--vector-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="n150_summary")
    parser.add_argument("--arch-name", type=str, default="wormhole_b0")
    args = parser.parse_args()

    sweep_fw = Path("/home/ubuntu/tt-metal/tests/sweep_framework")
    results_export = sweep_fw / "results_export"
    results_export.mkdir(parents=True, exist_ok=True)

    before_files = {p.name for p in results_export.glob("model_traced_*.json")}

    env = dict(os.environ)
    env["ARCH_NAME"] = args.arch_name

    cmd = [
        "/home/ubuntu/tt-metal/python_env/bin/python3",
        "sweeps_runner.py",
        "--module-name",
        "model_traced.matmul_model_traced",
        "--suite-name",
        "model_traced",
        "--vector-source",
        "file",
        "--file-path",
        str(args.vector_json),
        "--result-dest",
        "results_export",
        "--perf",
        "--summary",
        "--tag",
        args.tag,
    ]
    completed = subprocess.run(
        cmd,
        cwd=sweep_fw,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("sweeps_runner failed.\n" f"stdout={completed.stdout}\n" f"stderr={completed.stderr}")

    after_files = sorted(results_export.glob("model_traced_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = next((p for p in after_files if p.name not in before_files), None)
    if latest is None and after_files:
        latest = after_files[0]
    if latest is None:
        raise RuntimeError("No model_traced results file found after run")

    rows = json.loads(latest.read_text(encoding="utf-8"))
    pass_rows = [r for r in rows if r.get("status") == "pass"]
    fail_rows = [r for r in rows if r.get("status") != "pass"]
    e2e_vals = sorted(v for v in (_extract_e2e_ms(r) for r in rows) if v is not None)

    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vector_json": str(args.vector_json),
        "results_json": str(latest),
        "total_rows": len(rows),
        "pass_rows": len(pass_rows),
        "fail_rows": len(fail_rows),
        "summary": {
            "case_count": len(rows),
            "pass_rate": (len(pass_rows) / len(rows)) if rows else 0.0,
            "overall_p50_ms": _percentile(e2e_vals, 0.50),
            "overall_p95_ms": _percentile(e2e_vals, 0.95),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote sweep summary JSON -> {args.out_json}")
    print(f"pass_rate={payload['summary']['pass_rate']}")
    print(f"overall_p50_ms={payload['summary']['overall_p50_ms']}")
    print(f"overall_p95_ms={payload['summary']['overall_p95_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
