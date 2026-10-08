# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent, bounded stage attribution after the queued placement diagnostic."""

import argparse
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save, wait_for_sweep
from models.demos.qwen38_27b_qb2.tests.bounded_profile import CASES, SCOPE, VARIANTS, check_artifact_budget, collect


def run_capture(command, *, cwd, env, root, timeout=7200):
    """Own one process group; never kill or reset an unrelated hardware job."""
    check_artifact_budget(root)
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    started = time.monotonic()
    try:
        while process.poll() is None:
            check_artifact_budget(root)
            if time.monotonic() - started > timeout:
                raise TimeoutError("Bounded profile exceeded its capture/export deadline")
            time.sleep(2)
        check_artifact_budget(root)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    # The Blaze wrapper under task/source does not support --profile-ops.
    # Freeze the profiling-capable Metal wrapper beside this source snapshot.
    runner = args.source / "scripts/run_safe_pytest.sh"
    if "--profile-ops)" not in runner.read_text():
        raise ValueError("Frozen test runner lacks the --profile-ops option")
    files = [args.source / name for name in ("conftest.py", "pytest.ini", "scripts/run_safe_pytest.sh")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(
        state="cpu_validation",
        source=str(args.source),
        source_sha256=hashes,
        scope=SCOPE,
        promoted_to_model=False,
        p0_gate_passed=False,
        runs=[],
        hardware_lock="/tmp/tt-device.lock",
    )
    save(status_path, status)
    try:
        env = environment(args.task, args.source, args.weights)
        for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(key, None)
        subprocess.run(["/bin/bash", "-n", str(runner)], check=True, timeout=30)
        help_result = subprocess.run(
            [str(args.task / "python_env/bin/python"), "-m", "tracy", "--help"],
            cwd=args.source,
            env=env,
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )
        (args.results / "tracy-help.txt").write_text(help_result.stdout + help_result.stderr)
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
        wait_for_sweep(args.after_unit, args.after_results, status, status_path, receipt_names=("placement.json",))
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Queued source changed: {relative}")
        for length, batch in CASES:
            for variant in VARIANTS:
                directory = args.results / f"s{length}-b{batch}-{variant}"
                directory.mkdir()
                variant_env = dict(env)
                config = "precision_accurate_decode.json" if variant == "native" else "precision_single_step_gdn.json"
                variant_env.update(
                    QWEN_BOUNDED_LAYER_PROFILE="1",
                    QWEN_PROFILE_CONTEXT=str(length),
                    QWEN_PROFILE_BATCH=str(batch),
                    QWEN_PROFILE_RECURRENCE=variant,
                    QWEN_PROFILE_RECEIPT=str(directory / "profile.json"),
                    QWEN_PRECISION_CONFIG=str(model / "config" / config),
                    TT_METAL_PROFILER_DIR=str(directory / "tracy"),
                    TRACY_NO_WEB_SERVER="1",
                    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES="1",
                )
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=180",
                    "7200",
                    "/bin/bash",
                    str(runner),
                    "--profile-ops",
                    str(model / "tests/test_bounded_layer_profile.py"),
                    f"--rootdir={args.source}",
                    "-c",
                    str(args.source / "pytest.ini"),
                    "-vv",
                    "-s",
                    "--tb=short",
                    "--timeout=1200",
                    f"--junitxml={directory}/hardware.xml",
                ]
                row = dict(
                    input_tokens=length,
                    batch=batch,
                    recurrence=variant,
                    directory=str(directory),
                    command=command,
                    state="running_or_waiting_for_lock",
                )
                status["runs"].append(row)
                status.update(state="bounded_profile", active_case=[length, batch, variant])
                save(status_path, status)
                run_capture(command, cwd=args.source, env=variant_env, root=directory, timeout=7380)
                report = collect(directory, length, batch, variant)
                row.update(state="completed", measurements_complete=report["measurements_complete"])
                save(status_path, status)
        status.update(state="completed", cleanup_completed=True)
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "after-results"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit", required=True)
    run(parser.parse_args())
