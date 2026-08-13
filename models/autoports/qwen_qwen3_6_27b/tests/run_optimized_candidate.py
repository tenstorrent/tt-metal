# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run one optimized traced candidate and retain a compact machine-readable log."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "linear"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = Path(__file__).with_name("traced_synthetic_pcc.py")
    command = [
        sys.executable,
        str(runner),
        "--kind",
        args.kind,
        "--batch",
        str(args.batch),
        "--optimized",
        "--candidate",
        args.candidate,
        "--steps",
        str(args.steps),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[4],
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        combined = completed.stdout.splitlines() + completed.stderr.splitlines()
        markers = (
            "OPTIMIZED_POLICY",
            "_TRACED_SYNTHETIC_PCC",
            "_TRACED_SYNTHETIC_LATENCY",
            "TT_THROW",
            "TT_FATAL",
            "RuntimeError:",
            "critical",
        )
        retained = [line for line in combined if any(marker in line for marker in markers)]
        result = {
            "argv": command,
            "batch": args.batch,
            "candidate": args.candidate,
            "exit_status": completed.returncode,
            "kind": args.kind,
            "result": "pass" if completed.returncode == 0 else "fail",
            "retained_output": retained,
            "steps": args.steps,
        }
    except subprocess.TimeoutExpired as error:
        result = {
            "argv": command,
            "batch": args.batch,
            "candidate": args.candidate,
            "exit_status": 124,
            "kind": args.kind,
            "result": "timeout",
            "retained_output": [
                part
                for part in (
                    (error.stdout or ""),
                    (error.stderr or ""),
                )
                if part
            ],
            "steps": args.steps,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["result"] == "pass" else 1)


if __name__ == "__main__":
    main()
