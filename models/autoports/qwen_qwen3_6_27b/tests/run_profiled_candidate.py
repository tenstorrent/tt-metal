# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile one optimized candidate and retain exact machine-readable provenance."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "linear"), required=True)
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[4]
    if args.phase == "decode":
        runner = Path(__file__).with_name("traced_synthetic_pcc.py")
        runner_argv = [
            "--kind",
            args.kind,
            "--batch",
            str(args.batch),
            "--optimized",
            "--candidate",
            args.candidate,
            "--steps",
            "3",
        ]
    else:
        prefill_iterations = 5 if args.kind == "full" else 3
        runner = Path(__file__).with_name(
            "full_attention_synthetic_pcc.py" if args.kind == "full" else "linear_attention_synthetic_pcc.py"
        )
        runner_argv = [
            "--mode",
            "prefill",
            "--sequence",
            "33" if args.kind == "full" else "5",
            "--batch",
            str(args.batch),
            "--optimized",
            "--candidate",
            args.candidate,
            "--iterations",
            str(prefill_iterations),
        ]
    artifact = (repo / args.artifact_dir).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    profile_argv = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-o",
        str(artifact),
        str(runner),
        *runner_argv,
    ]
    profile = subprocess.run(
        profile_argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    reports = sorted(artifact.glob("reports/*/ops_perf_results_*.csv"))
    if profile.returncode != 0 or len(reports) != 1:
        result = {
            "kind": args.kind,
            "phase": args.phase,
            "batch": args.batch,
            "candidate": args.candidate,
            "profile_argv": profile_argv,
            "profile_exit_status": profile.returncode,
            "report_candidates": [str(path) for path in reports],
            "retained_output": profile.stdout.splitlines()[-80:],
        }
        (artifact / "profile_run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise SystemExit(profile.returncode or 1)

    signpost = "PERF_DECODE" if args.phase == "decode" else "PERF_PREFILL"
    perf_argv = [
        str(repo / "python_env/bin/tt-perf-report"),
        str(reports[0]),
        "--start-signpost",
        signpost,
        "--end-signpost",
        f"{signpost}_END",
        "--no-color",
        "--csv",
        str(artifact / "perf.csv"),
        "--summary-file",
        str(artifact / "summary"),
    ]
    perf = subprocess.run(
        perf_argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = {
        "kind": args.kind,
        "phase": args.phase,
        "batch": args.batch,
        "candidate": args.candidate,
        "sequence": 1 if args.phase == "decode" else (33 if args.kind == "full" else 5),
        "iterations": (2 if args.phase == "decode" else (5 if args.kind == "full" else 3)),
        "profile_argv": profile_argv,
        "profile_exit_status": profile.returncode,
        "raw_report": str(reports[0]),
        "perf_argv": perf_argv,
        "perf_exit_status": perf.returncode,
        "profile_output": [
            line
            for line in profile.stdout.splitlines()
            if "OPTIMIZED_POLICY" in line or "_PCC" in line or "_LATENCY" in line or "ops_perf_results_" in line
        ],
        "perf_output": perf.stdout.splitlines(),
    }
    (artifact / "profile_run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if perf.returncode != 0:
        raise SystemExit(perf.returncode)


if __name__ == "__main__":
    main()
