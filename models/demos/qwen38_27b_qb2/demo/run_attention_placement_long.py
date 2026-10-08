# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded BFP8 attention diagnostic, serialized by the existing device lock."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / name for name in ("conftest.py", "pytest.ini")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(
        state="cpu_validation",
        source=str(args.source),
        source_sha256=hashes,
        precision_change=False,
        hardware_lock="/tmp/tt-device.lock",
        promoted_to_model=False,
    )
    save(status_path, status)
    try:
        env = environment(args.task, args.source, args.weights)
        # Never inherit a simulator selection or a previous source overlay.
        for name in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(name, None)
        env.update(
            QWEN_ATTENTION_PLACEMENT_LONG="1", QWEN_ATTENTION_PLACEMENT_RECEIPT=str(args.results / "placement.json")
        )
        subprocess.run(
            [
                str(args.task / "python_env/bin/python"),
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
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Queued source changed: {relative}")
        # The wrapper acquires flock before any hardware action. Its dispatch
        # deadline is five seconds; pytest's 40-minute timer starts after flock.
        # The enclosing three-hour budget includes waiting behind the live run.
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180",
            "10800",
            "/bin/bash",
            str(args.task / "source/scripts/run_safe_pytest.sh"),
            str(model / "tests/test_attention_placement_long.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=2400",
            f"--junitxml={args.results}/hardware.xml",
        ]
        status.update(state="waiting_for_device_lock_or_running", command=command)
        save(status_path, status)
        subprocess.run(command, cwd=args.source, env=env, check=True, timeout=11100)
        receipt = json.loads((args.results / "placement.json").read_text())
        if (
            receipt.get("state") != "completed"
            or receipt.get("passed") is not True
            or receipt.get("cleanup_completed") is not True
        ):
            raise RuntimeError("Diagnostic lacks complete results and clean device close")
        status.update(
            state="completed",
            cleanup_completed=True,
            note="Individual numerical failures remain excluded; completed diagnostic is not model qualification",
        )
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
