# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Queue a layer diagnostic after CPU work, only if the head GPQA control misses."""

import argparse
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save


def cpu_predecessor_ready(properties, receipt, invocation):
    if properties.get("InvocationID") and properties["InvocationID"] != invocation:
        raise ValueError("CPU predecessor service was replaced")
    if (
        properties.get("LoadState") not in ("loaded", "not-found")
        or properties.get("MainPID") != "0"
        or properties.get("ActiveState") not in ("inactive", "failed")
    ):
        return False
    if properties.get("LoadState") == "loaded" and properties.get("Result") != "success":
        raise ValueError("CPU predecessor service failed")
    if not receipt or receipt.get("state") != "completed":
        raise ValueError("CPU predecessor lacks a completed receipt")
    stages = [row for row in receipt.get("stages", []) if row.get("name") == "hf-head-reference"]
    if len(stages) != 1 or stages[0].get("state") != "completed":
        raise ValueError("CPU HF reference did not complete")
    return True


def needs_diagnostic(summary):
    result = summary["gpqa_result"]
    if result.get("completed_samples") != 198 or result.get("dataset_samples") != 198:
        raise ValueError("Head control must finish the complete GPQA set")
    correct = result.get("correct")
    if type(correct) is not int or not 0 <= correct <= 198 or result.get("passed") != (correct >= 177):
        raise ValueError("Head control score disagrees with the unchanged release gate")
    return correct < 177


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
                raise TimeoutError("CPU predecessor wait exceeded its bound")
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
            if cpu_predecessor_ready(properties, receipt, args.predecessor_invocation):
                break
            time.sleep(30)
        summary = json.loads(args.gpqa_summary.read_text())
        if not needs_diagnostic(summary):
            state.update(
                state="completed",
                hardware_opened=False,
                reason="Head control met the full GPQA gate; skip additional accuracy diagnosis",
            )
            return
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
        precision = model / "config/precision_accurate_decode_bfp8_head.json"
        env.update(
            QWEN_HF_LAYER_REFERENCE="1",
            QWEN_HF_WEIGHTS=str(args.weights),
            QWEN_HF_QUALIFICATION=str(args.qualification),
            QWEN_HF_PRECISION=str(precision),
            QWEN_HF_REFERENCE=str(args.reference),
            QWEN_HF_OUTPUT=str(args.results / "comparison"),
            QWEN_PRECISION_CONFIG=str(precision),
            QWEN_DECODE_BUCKETS="1",
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_hf_layer_reference.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--timeout=2400",
            f"--junitxml={args.results}/hardware.xml",
        ]
        state.update(state="running_or_waiting_for_hardware_lock", command=command, hardware_opened=None)
        save(status_path, state)
        from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture

        run_capture(command, cwd=args.source, env=env, root=args.results, timeout=3000)
        report = json.loads((args.results / "comparison/progress.json").read_text())
        suites = ET.parse(args.results / "hardware.xml").getroot().findall(".//testsuite")
        if (
            report.get("state") != "completed"
            or report.get("cleanup_completed") is not True
            or len(report.get("steps", [])) != 8
            or sum(int(s.get("tests", 0)) for s in suites) != 1
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise ValueError("Layer diagnostic did not complete cleanly")
        state.update(state="completed", hardware_opened=True, cleanup_completed=True, is_gpqa_score=False)
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
    for name in (
        "task",
        "source",
        "weights",
        "results",
        "manifest",
        "qualification",
        "reference",
        "predecessor-receipt",
        "gpqa-summary",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--predecessor-unit", required=True)
    parser.add_argument("--predecessor-invocation", required=True)
    parser.add_argument("--wait-timeout", type=float, default=72000)
    run(parser.parse_args())
