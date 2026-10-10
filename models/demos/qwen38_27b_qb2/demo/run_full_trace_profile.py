# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded persistent full-model trace calibration after qualification/transport."""

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import ARTIFACT_BUDGET, CASES, SCOPE, collect


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / name for name in ("conftest.py", "pytest.ini", "scripts/run_safe_pytest.sh")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json", ".sh"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(
        state="cpu_validation",
        source=str(args.source),
        source_sha256=hashes,
        scope=SCOPE,
        p0_gate_passed=False,
        runs=[],
        hardware_lock="/tmp/tt-device.lock",
    )
    save(status_path, status)
    try:
        env = environment(args.task, args.source, args.weights)
        for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(key, None)
        env.update(
            QWEN_PRECISION_CONFIG=str(model / "config/precision_single_step_shared_qk_bfp8_all.json"),
            TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="8192",
        )
        subprocess.run(["/bin/bash", "-n", str(args.source / "scripts/run_safe_pytest.sh")], check=True, timeout=30)
        subprocess.run(
            [str(args.task / "python_env/bin/python"), "-m", "tracy", "--help"],
            cwd=args.source,
            env=env,
            check=True,
            timeout=60,
            capture_output=True,
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
                "-o",
                "addopts=",
                "-q",
                f"--junitxml={args.results}/unit.xml",
            ],
            cwd=args.source,
            env=env,
            check=True,
            timeout=600,
        )
        deadline = time.monotonic() + 43200
        while True:
            result = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    args.after_unit,
                    "-p",
                    "LoadState",
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            status.update(state="waiting_for_delivery_job", dependency=props)
            save(status_path, status)
            if props.get("LoadState") == "not-found" or (
                props.get("ActiveState") in ("inactive", "failed") and props.get("MainPID") == "0"
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Dependency is still live after twelve hours; do not interrupt it")
            time.sleep(20)
        for name, digest in hashes.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Frozen full-profile source changed: {name}")
        for length, batch in CASES:
            directory = args.results / f"s{length}-b{batch}"
            directory.mkdir()
            case_env = dict(
                env,
                QWEN_FULL_TRACE_PROFILE="1",
                QWEN_PROFILE_CONTEXT=str(length),
                QWEN_PROFILE_BATCH=str(batch),
                QWEN_PROFILE_RECEIPT=str(directory / "profile.json"),
                TT_METAL_PROFILER_DIR=str(directory / "tracy"),
                TRACY_NO_WEB_SERVER="1",
                TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES="1",
            )
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=180",
                "5400",
                "/bin/bash",
                str(args.source / "scripts/run_safe_pytest.sh"),
                "--profile-ops",
                str(model / "tests/test_full_trace_profile.py"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-vv",
                "-s",
                "--tb=short",
                "--timeout=3600",
                f"--junitxml={directory}/hardware.xml",
            ]
            row = dict(
                input_tokens=length,
                batch=batch,
                directory=str(directory),
                command=command,
                state="profiling_or_waiting_for_lock",
            )
            status["runs"].append(row)
            status.update(state="profiling", active_case=[length, batch])
            save(status_path, status)
            run_capture(
                command, cwd=args.source, env=case_env, root=directory, timeout=5580, artifact_budget=ARTIFACT_BUDGET
            )
            report = collect(directory)
            row.update(
                state="completed",
                measurements_complete=report["measurements_complete"],
                full_trace_reconciliation_passed=report["full_trace_reconciliation_passed"],
                comparisons=report["comparisons"],
            )
            save(status_path, status)
        status.update(
            state="completed",
            cleanup_completed=True,
            full_trace_reconciliation_passed=all(r["full_trace_reconciliation_passed"] for r in status["runs"]),
        )
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit", required=True)
    run(parser.parse_args())
