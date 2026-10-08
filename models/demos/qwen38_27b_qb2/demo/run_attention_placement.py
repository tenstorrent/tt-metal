# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the placement diagnostic persistently after the capacity experiment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save, wait_for_sweep


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / name for name in ("conftest.py", "pytest.ini")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(state="queued", source=str(args.source), source_sha256=hashes, precision_change=False)
    save(status_path, status)
    try:
        wait_for_sweep(args.after_unit, args.after_results, status, status_path, variants=("capacity",))
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Queued source changed: {relative}")
        env = environment(args.task, args.source, args.weights)
        env.update(QWEN_ATTENTION_PLACEMENT="1", QWEN_ATTENTION_PLACEMENT_RECEIPT=str(args.results / "placement.json"))
        python = str(args.task / "python_env/bin/python")
        status["state"] = "cpu_validation"
        save(status_path, status)
        subprocess.run(
            [
                python,
                "-m",
                "pytest",
                str(model / "tests/unit"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-q",
                f"--junitxml={args.results}/unit.xml",
            ],
            cwd=args.source,
            env=env,
            check=True,
            timeout=600,
        )
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180",
            "5400",
            "/bin/bash",
            str(args.task / "source/scripts/run_safe_pytest.sh"),
            str(model / "tests/test_attention_placement.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=5220",
            f"--junitxml={args.results}/hardware.xml",
        ]
        status.update(state="hardware_diagnostic", command=command)
        save(status_path, status)
        subprocess.run(command, cwd=args.source, env=env, check=True, timeout=5700)
        receipt = json.loads((args.results / "placement.json").read_text())
        if (
            receipt.get("passed") is not True
            or receipt.get("state") != "completed"
            or receipt.get("cleanup_completed") is not True
        ):
            raise RuntimeError("Placement process lacks a completed diagnostic and clean-device receipt")
        status.update(state="completed", promoted_to_model=False)
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "after-results"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--after-unit", required=True)
    run(parser.parse_args())
