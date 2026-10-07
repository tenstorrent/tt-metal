# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded sweep controller: restart in a fresh process after a clean allocator OOM."""

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.sweep_recovery import can_restart
from models.demos.qwen38_27b_qb2.tests.sweep_report import render, save_report


def run(args):
    root = args.results
    plan = json.loads((root / "sweep.json").read_text())
    if plan["state"] != "queued":
        raise ValueError("Use a newly initialized sweep directory")
    receipts = [str(path.resolve()) for path in args.resume]
    deadline = time.monotonic() + args.timeout
    for attempt in range(1, len(plan["cells"]) + 2):
        directory = root / f"attempt-{attempt:02d}"
        directory.mkdir()
        save_report(copy.deepcopy(plan), directory)
        remaining = int(deadline - time.monotonic())
        if remaining <= 300:
            raise TimeoutError("Not enough time remains to load a fresh model for this sweep")
        environment = dict(os.environ, QWEN_SWEEP_RESULTS=str(directory), QWEN_SWEEP_RESUME_FROM=json.dumps(receipts))
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180",
            str(remaining),
            "/bin/bash",
            str(args.task / "source/scripts/run_safe_pytest.sh"),
            str(args.source / "models/demos/qwen38_27b_qb2/tests/test_galaxy_perf_sweep.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            f"--timeout={remaining - 120}",
            f"--junitxml={directory}/hardware.xml",
        ]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"SWEEP_ATTEMPT_BEGIN {directory}", flush=True)
        result = subprocess.run(command, env=environment, check=False)
        measured = json.loads((directory / "sweep.json").read_text())
        measured["attempt_returncode"] = result.returncode
        measured["latest_attempt"] = str(directory)
        # The per-attempt receipt is left intact. This root copy is the current aggregate view.
        save_report(measured, root)
        render(measured, root)
        if result.returncode == 0:
            if (
                measured.get("cleanup_completed") is not True
                or measured["state"] not in ("completed", "completed_with_oom")
                or any(
                    cell["status"] not in ("completed", "oom", "capacity_guard", "implementation_guard")
                    for cell in measured["cells"]
                )
            ):
                raise RuntimeError("Successful process lacks a completed sweep and device cleanup receipt")
            return
        if not can_restart(measured, result.returncode):
            raise RuntimeError(f"Sweep attempt failed with code {result.returncode}; refusing automatic recovery")
        receipts.append(str(directory / "sweep.json"))
        if all(
            cell["status"] in ("completed", "oom", "capacity_guard", "implementation_guard")
            for cell in measured["cells"]
        ):
            measured["state"] = "completed_with_oom"
            save_report(measured, root)
            render(measured, root)
            return
        print("SWEEP_ALLOCATION_FAILURE: recorded failed cell; remaining cells will use a fresh process", flush=True)
    raise RuntimeError("Sweep recovery exceeded the number of planned cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--resume", action="append", default=[], type=Path)
    parser.add_argument("--timeout", type=int, default=21600)
    run(parser.parse_args())
