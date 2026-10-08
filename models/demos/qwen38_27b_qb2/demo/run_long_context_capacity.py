# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent fixed-precision capacity experiment after a completed hardware sweep."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def save(path, data):
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def wait_for_sweep(unit, results, status, status_path):
    """A quiet log or observation timeout is not evidence that hardware is idle."""
    while True:
        observed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
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
        properties = dict(line.split("=", 1) for line in observed.stdout.splitlines() if "=" in line)
        status.update(state="waiting_for_sweep", dependency=properties)
        save(status_path, status)
        missing = properties.get("LoadState") == "not-found"
        terminal = properties.get("ActiveState") in ("inactive", "failed") or properties.get("SubState") == "exited"
        if missing or (terminal and properties.get("MainPID") == "0"):
            # Successful transient units may be garbage-collected. Their missing
            # handle is only sufficient together with both clean terminal receipts.
            if not missing and properties.get("Result") != "success":
                raise RuntimeError(f"Dependency sweep did not exit successfully: {properties}")
            for variant in ("native", "single-step"):
                receipt = json.loads((results / variant / "sweep.json").read_text())
                if (
                    receipt.get("state") not in ("completed", "completed_with_oom")
                    or receipt.get("cleanup_completed") is not True
                ):
                    raise RuntimeError(f"Dependency {variant} has no terminal sweep and clean-device receipt")
            return
        time.sleep(20)


def environment(task, source, weights):
    env = {key: value for key, value in os.environ.items() if not key.startswith("QWEN_")}
    for key in (
        "TT_METAL_SLOW_DISPATCH_MODE",
        "TT_METAL_ALLOCATOR_MODE_HYBRID",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_PROFILER_MID_RUN_DUMP",
        "TT_METAL_PROFILER_CPP_POST_PROCESS",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{task}/python_env/bin:" + env.get("PATH", "/usr/bin:/bin"),
        TT_METAL_HOME=str(task / "metal"),
        PYTHONPATH=f"{source}:{task}/metal:{task}/metal/tools",
        LD_LIBRARY_PATH=f"{task}/metal-install/lib:{task}/metal-build/lib",
        TT_METAL_CACHE=str(task / "jit-cache-metal-galaxy"),
        MPLCONFIGDIR=str(task / "matplotlib-cache"),
        MODEL_WEIGHTS_DIR=str(weights),
        ARCH_NAME="blackhole",
        OMP_NUM_THREADS="8",
        PYTHONUNBUFFERED="1",
        QWEN_PRECISION_CONFIG=str(source / "models/demos/qwen38_27b_qb2/config/precision_accurate_decode.json"),
        QWEN_DECODE_BUCKETS="0",
        QWEN_GALAXY_SWEEP="1",
        QWEN_COMPACT_DECODE_RESIDUAL="1",
        QWEN_COMPACT_DECODE_MLP="1",
        QWEN_COMPACT_DECODE_ATTENTION="1",
        QWEN_BATCHED_DECODE_ROPE="1",
        QWEN_BATCHED_PREFILL="1",
        QWEN_PREFILL_RESIDUAL_LAYOUT="sharded_replicated_norm",
        QWEN_PREFILL_BATCHED_HEAD="1",
        QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD="1",
        QWEN_PREFILL_STARTUP_WARMUP="1",
        QWEN_PREFILL_MAX_BATCH_TOKENS="32768",
    )
    return env


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    files = [args.source / name for name in ("conftest.py", "pytest.ini")]
    model_source = args.source / "models/demos/qwen38_27b_qb2"
    files.extend(path for path in model_source.rglob("*") if path.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    status = dict(state="queued", source=str(args.source), source_sha256=hashes, precision_change=False)
    save(status_path, status)
    try:
        if args.after_unit:
            wait_for_sweep(args.after_unit, args.after_results, status, status_path)
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Queued source changed: {relative}")
        env = environment(args.task, args.source, args.weights)
        python = str(args.task / "python_env/bin/python")
        status.update(state="cpu_validation")
        save(status_path, status)
        subprocess.run(
            [
                python,
                "-m",
                "pytest",
                str(model_source / "tests/unit"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-q",
                f"--junitxml={args.results}/unit.xml",
            ],
            env=env,
            cwd=args.source,
            check=True,
            timeout=600,
        )
        # Import only host-side planning after dependency shutdown. No device opens here.
        from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, render, save_report

        plan = make_plan(1, batches=(8, 16, 32), input_lengths=(32768, 131072, 262016), max_pool_tokens=2359296)
        order = ((32768, 16), (32768, 32), (131072, 16), (262016, 8))
        cells = {(cell["input_tokens"], cell["batch_per_replica"]): cell for cell in plan["cells"]}
        plan["cells"] = [cells[key] for key in order]
        plan["recurrence_variant"] = "native"
        plan[
            "qualification_scope"
        ] = "Fixed-precision capacity and repeatability experiment; changed prefill chunking is not reference-eval qualified"
        sweep = args.results / "capacity"
        save_report(plan, sweep)
        render(plan, sweep)
        command = [
            python,
            "-m",
            "models.demos.qwen38_27b_qb2.demo.run_sweep_attempts",
            "--task",
            str(args.task),
            "--source",
            str(args.source),
            "--results",
            str(sweep),
            "--timeout",
            "21600",
        ]
        status.update(state="hardware_sweep", command=command)
        save(status_path, status)
        subprocess.run(command, env=env, cwd=args.source, check=True, timeout=22000)
        measured = json.loads((sweep / "sweep.json").read_text())
        status.update(state="completed", sweep_state=measured["state"], cleanup_completed=measured["cleanup_completed"])
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit")
    parser.add_argument("--after-results", type=Path)
    args = parser.parse_args()
    if bool(args.after_unit) != bool(args.after_results):
        parser.error("--after-unit and --after-results must be supplied together")
    run(args)
