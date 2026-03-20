#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _delta_pct(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or base == 0:
        return None
    return ((cand / base) - 1.0) * 100.0


def _run_command(command_template: str, out_json: Path, env: dict[str, str]) -> None:
    command = command_template.format(out_json=str(out_json))
    result = subprocess.run(
        shlex.split(command),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed.\n" f"command={command}\n" f"stdout={result.stdout}\n" f"stderr={result.stderr}"
        )


def _collect_run(run_json: Path) -> dict[str, Any]:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    return payload.get("summary", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Alternating A/B orchestration for N150 matmul kernelbench outputs.")
    parser.add_argument(
        "--baseline-cmd-template",
        required=True,
        help="Command template with '{out_json}' placeholder for baseline run output path.",
    )
    parser.add_argument(
        "--candidate-cmd-template",
        required=True,
        help="Command template with '{out_json}' placeholder for candidate run output path.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Number of baseline/candidate pairs.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/deepseek_oriented/matmul_n150_alternating_ab_last.json"
        ),
    )
    parser.add_argument("--p50-key", type=str, default="overall_e2e_p50_ms")
    parser.add_argument("--p95-key", type=str, default="overall_e2e_p95_ms")
    parser.add_argument("--kernel-p50-key", type=str, default="overall_kernel_p50_ns")
    parser.add_argument("--kernel-p95-key", type=str, default="overall_kernel_p95_ns")
    args = parser.parse_args()

    if "{out_json}" not in args.baseline_cmd_template or "{out_json}" not in args.candidate_cmd_template:
        raise ValueError("Both command templates must include '{out_json}' placeholder.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")

    out_dir = args.out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)

    baseline_runs: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []

    # Alternate baseline/candidate in fresh processes.
    for round_idx in range(1, args.rounds + 1):
        b_json = out_dir / f"{args.out_json.stem}_baseline_r{round_idx}.json"
        _run_command(args.baseline_cmd_template, b_json, env)
        b_summary = _collect_run(b_json)
        baseline_runs.append({"round": round_idx, "out_json": str(b_json), "summary": b_summary})
        schedule.append({"run_type": "baseline", "round": round_idx, "out_json": str(b_json)})

        c_json = out_dir / f"{args.out_json.stem}_candidate_r{round_idx}.json"
        _run_command(args.candidate_cmd_template, c_json, env)
        c_summary = _collect_run(c_json)
        candidate_runs.append({"round": round_idx, "out_json": str(c_json), "summary": c_summary})
        schedule.append({"run_type": "candidate", "round": round_idx, "out_json": str(c_json)})

    def extract(runs: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r["summary"][key]) for r in runs if key in r["summary"] and r["summary"][key] is not None]

    b_p50 = extract(baseline_runs, args.p50_key)
    b_p95 = extract(baseline_runs, args.p95_key)
    c_p50 = extract(candidate_runs, args.p50_key)
    c_p95 = extract(candidate_runs, args.p95_key)

    b_kp50 = extract(baseline_runs, args.kernel_p50_key)
    b_kp95 = extract(baseline_runs, args.kernel_p95_key)
    c_kp50 = extract(candidate_runs, args.kernel_p50_key)
    c_kp95 = extract(candidate_runs, args.kernel_p95_key)

    mean_b_p50 = _mean(b_p50)
    mean_b_p95 = _mean(b_p95)
    mean_c_p50 = _mean(c_p50)
    mean_c_p95 = _mean(c_p95)

    delta_p50 = _delta_pct(mean_b_p50, mean_c_p50)
    delta_p95 = _delta_pct(mean_b_p95, mean_c_p95)

    strict_merge_gate_pass = delta_p50 is not None and delta_p95 is not None and delta_p50 < 0.0 and delta_p95 < 0.0

    payload = {
        "rounds": args.rounds,
        "baseline_cmd_template": args.baseline_cmd_template,
        "candidate_cmd_template": args.candidate_cmd_template,
        "schedule": schedule,
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "aggregate": {
            "baseline": {
                f"{args.p50_key}_mean": mean_b_p50,
                f"{args.p50_key}_stdev": _stdev(b_p50),
                f"{args.p95_key}_mean": mean_b_p95,
                f"{args.p95_key}_stdev": _stdev(b_p95),
                f"{args.kernel_p50_key}_mean": _mean(b_kp50),
                f"{args.kernel_p95_key}_mean": _mean(b_kp95),
            },
            "candidate": {
                f"{args.p50_key}_mean": mean_c_p50,
                f"{args.p50_key}_stdev": _stdev(c_p50),
                f"{args.p95_key}_mean": mean_c_p95,
                f"{args.p95_key}_stdev": _stdev(c_p95),
                f"{args.kernel_p50_key}_mean": _mean(c_kp50),
                f"{args.kernel_p95_key}_mean": _mean(c_kp95),
            },
            "candidate_vs_baseline_delta_pct": {
                args.p50_key: delta_p50,
                args.p95_key: delta_p95,
                args.kernel_p50_key: _delta_pct(_mean(b_kp50), _mean(c_kp50)),
                args.kernel_p95_key: _delta_pct(_mean(b_kp95), _mean(c_kp95)),
            },
            "strict_merge_gate_pass": strict_merge_gate_pass,
        },
    }

    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote A/B report -> {args.out_json}")
    print(f"strict_merge_gate_pass={strict_merge_gate_pass}")
    print(f"delta {args.p50_key}={delta_p50}")
    print(f"delta {args.p95_key}={delta_p95}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
