# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Retain a like-for-like functional or optimized prefill measurement."""

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
    parser.add_argument("--implementation", choices=("functional", "optimized"), required=True)
    parser.add_argument("--candidate", default="default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sequence = 33 if args.kind == "full" else 5
    iterations = 5 if args.kind == "full" else 3
    runner = Path(__file__).with_name(
        "full_attention_synthetic_pcc.py" if args.kind == "full" else "linear_attention_synthetic_pcc.py"
    )
    command = [
        sys.executable,
        str(runner),
        "--mode",
        "prefill",
        "--sequence",
        str(sequence),
        "--batch",
        str(args.batch),
        "--iterations",
        str(iterations),
    ]
    if args.implementation == "optimized":
        command.extend(("--optimized", "--candidate", args.candidate))

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[4],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    markers = (
        "FALLBACK_AUDIT",
        "OPTIMIZED_POLICY",
        "_SYNTHETIC_PCC",
        "_SYNTHETIC_LATENCY",
        "TT_THROW",
        "TT_FATAL",
        "RuntimeError:",
        "critical",
    )
    result = {
        "argv": command,
        "batch": args.batch,
        "candidate": args.candidate if args.implementation == "optimized" else "functional",
        "exit_status": completed.returncode,
        "implementation": args.implementation,
        "iterations": iterations,
        "kind": args.kind,
        "result": "pass" if completed.returncode == 0 else "fail",
        "retained_output": [
            line for line in completed.stdout.splitlines() if any(marker in line for marker in markers)
        ],
        "sequence": sequence,
        "timing_contract": (
            "one untimed warmup, then wall-clock each iteration with ttnn.synchronize_device; "
            "reported latency is the median"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if completed.returncode == 0 else 1)


if __name__ == "__main__":
    main()
