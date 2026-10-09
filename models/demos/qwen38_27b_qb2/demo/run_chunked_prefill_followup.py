# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize the chunked-state diagnostic after accuracy and CPU reference work."""

import argparse
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save


def predecessor_ready(properties, receipt, invocation):
    if properties.get("InvocationID") and properties["InvocationID"] != invocation:
        raise ValueError("Predecessor invocation changed")
    if (
        properties.get("LoadState") not in ("loaded", "not-found")
        or properties.get("MainPID") != "0"
        or properties.get("ActiveState") not in ("inactive", "failed")
    ):
        return False
    if properties.get("LoadState") == "loaded" and properties.get("Result") != "success":
        raise ValueError("Predecessor service failed")
    if not receipt or receipt.get("state") != "completed":
        raise ValueError("Predecessor lacks a completed receipt")
    if receipt.get("hardware_opened") is not False and receipt.get("cleanup_completed") is not True:
        raise ValueError("Predecessor hardware cleanup is unproven")
    return True


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    manifest = json.loads(args.manifest.read_text())
    state = dict(
        state="waiting",
        source_sha256=manifest,
        hardware_opened=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        predecessor_unit=args.predecessor_unit,
        predecessor_invocation=args.predecessor_invocation,
    )
    save(status_path, state)
    try:
        deadline = time.monotonic() + args.wait_timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Predecessor wait exceeded its bound")
            try:
                output = subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.predecessor_unit,
                        "-p",
                        "InvocationID",
                        "-p",
                        "LoadState",
                        "-p",
                        "ActiveState",
                        "-p",
                        "MainPID",
                        "-p",
                        "Result",
                    ],
                    text=True,
                    timeout=15,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                state.update(observation_error=type(error).__name__)
                save(status_path, state)
                time.sleep(30)
                continue
            properties = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
            receipt = json.loads(args.predecessor_receipt.read_text()) if args.predecessor_receipt.exists() else None
            state.update(predecessor=properties)
            save(status_path, state)
            if predecessor_ready(properties, receipt, args.predecessor_invocation):
                break
            time.sleep(30)
        for relative, expected in manifest.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        model = args.source / "models/demos/qwen38_27b_qb2"
        env = environment(args.task, args.source, args.weights)
        for name in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_KERNEL_PATH",
            "TT_METAL_DISABLE_SFPLOADMACRO",
            "TT_VISIBLE_DEVICES",
        ):
            env.pop(name, None)
        env.update(
            QWEN_CHUNKED_STATE_TEST="1",
            QWEN_CHUNKED_QUALIFICATION=str(args.qualification),
            QWEN_CHUNKED_PRECISION=str(model / "config/precision_accurate_decode.json"),
            QWEN_CHUNKED_OUTPUT=str(args.results / "comparison"),
            QWEN_DECODE_BUCKETS="1",
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_chunked_prefill_hardware.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--timeout=3600",
            f"--junitxml={args.results}/hardware.xml",
        ]
        state.update(state="running_or_waiting_for_hardware_lock", command=command, hardware_opened=None)
        save(status_path, state)
        from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture

        run_capture(command, cwd=args.source, env=env, root=args.results, timeout=4200)
        report = json.loads((args.results / "comparison/progress.json").read_text())
        suites = ET.parse(args.results / "hardware.xml").getroot().findall(".//testsuite")
        if (
            report.get("state") != "completed"
            or report.get("cleanup_completed") is not True
            or report.get("passed") is not True
            or len(report.get("comparisons", [])) != 11
            or sum(int(s.get("tests", 0)) for s in suites) != 1
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise ValueError("Chunked-state diagnostic did not pass cleanly")
        state.update(
            state="completed",
            hardware_opened=True,
            cleanup_completed=True,
            passed=True,
            plugin_scheduler_qualified=False,
            sampler_qualified=False,
            model_capability_enabled=False,
        )
    except BaseException as error:
        state.update(state="failed", error=type(error).__name__, detail=str(error)[:2000])
        raise
    finally:
        comparison = args.results / "comparison/progress.json"
        if comparison.exists():
            observed = json.loads(comparison.read_text())
            state.update(
                hardware_opened=observed.get("hardware_opened"),
                cleanup_completed=observed.get("cleanup_completed", False),
            )
        save(status_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "manifest", "qualification", "predecessor-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--predecessor-unit", required=True)
    parser.add_argument("--predecessor-invocation", required=True)
    parser.add_argument("--wait-timeout", type=float, default=79200)
    run(parser.parse_args())
