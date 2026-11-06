#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed until lossless full-phase completion evidence is available."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--action", choices=("check", "collect"), required=True)
    args = parser.parse_args()
    identity = json.loads((args.run_dir / f"perf-b{args.concurrency}-server.json").read_text())
    failure = {
        "status": "unavailable",
        "action": args.action,
        "concurrency": args.concurrency,
        "server_pid": identity["pid"],
        "reason": "Pinned stock plugin has no lossless request/step completion export after host sampling and final token handling. Adapter return/enqueue timings cannot satisfy full_phase_wall_time.",
        "diagnosis": "../setup/AUTODEBUG.md",
        "plugin_runtime_change_authorized": False,
        "device_profiler_enabled": False,
    }
    (args.run_dir / f"roofline-b{args.concurrency}-unavailable.json").write_text(json.dumps(failure, indent=2) + "\n")
    raise SystemExit(failure["reason"])


if __name__ == "__main__":
    main()
