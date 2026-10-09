# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audit a stopped optional-capacity experiment before a separate resetting job."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.overnight_plan import optional_capacity_result


def validate(properties, queue, sweep, invocation):
    if (
        properties.get("InvocationID") != invocation
        or properties.get("MainPID") != "0"
        or properties.get("ActiveState") != "failed"
        or properties.get("Result") != "exit-code"
        or properties.get("ExecMainStatus") != "1"
    ):
        raise ValueError("Require the exact failed, stopped predecessor with ordinary exit code 1")
    stages = queue.get("stages", [])
    if (
        queue.get("state") != "failed"
        or queue.get("active_stage") != "bfp8-budget64k"
        or not stages
        or stages[-1].get("name") != "bfp8-budget64k"
        or stages[-1].get("state") != "failed"
        or any(row.get("state") != "completed" for row in stages[:-1])
    ):
        raise ValueError("Require completed preceding stages and only the optional 64K failure")
    if not optional_capacity_result("bfp8-budget64k", sweep, 1):
        raise ValueError("Require a proven allocator limit with clean device closure")


def run(args):
    assert not args.output.exists(), "Preserve prior recovery records"
    raw = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            args.previous_unit,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
            "-p",
            "InvocationID",
        ],
        text=True,
        timeout=30,
    )
    properties = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    queue = json.loads(args.queue.read_text())
    sweep = json.loads(args.sweep.read_text())
    validate(properties, queue, sweep, args.previous_invocation)
    with open("/tmp/tt-device.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = dict(
            state="completed",
            cleanup_completed=True,
            hardware_health_proven=False,
            device_reset_required=True,
            passed_hardware_test=False,
            physical_devices_accessed=False,
            invocation=os.environ.get("INVOCATION_ID"),
            previous_unit=args.previous_unit,
            previous=properties,
            reason="Optional 64K prefill budget exhausted DRAM; benchmark confirms clean closure; next job resets under lock",
            sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.queue, args.sweep)},
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "sweep", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-unit", required=True)
    parser.add_argument("--previous-invocation", required=True)
    run(parser.parse_args())
