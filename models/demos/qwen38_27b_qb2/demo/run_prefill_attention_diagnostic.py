# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent prefill numerical diagnosis after the prioritized decode experiments."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import predecessor_ready
from models.demos.qwen38_27b_qb2.tests.prefill_attention_diagnostic import validate_report


def run(args):
    args.output.mkdir()
    queue = args.output / "queue.json"
    status = dict(
        state="waiting",
        cleanup_completed=False,
        hardware_started=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        promoted_to_serving=False,
        predecessor_unit=args.after_unit,
        predecessor_invocation=args.after_invocation,
        hardware_lock="/tmp/tt-device.lock",
        started_at=time.time(),
    )

    def terminate(signum, frame):
        raise InterruptedError(f"Prefill numerical diagnostic queue received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(queue, status)
    try:
        deadline = time.monotonic() + 24 * 3600
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Predecessor remains live; no restart or hardware takeover")
            try:
                raw = subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.after_unit,
                        "-p",
                        "MainPID",
                        "-p",
                        "ActiveState",
                        "-p",
                        "LoadState",
                        "-p",
                        "Result",
                        "-p",
                        "InvocationID",
                    ],
                    text=True,
                    timeout=20,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                status["observation_error"] = type(error).__name__
                save(queue, status)
                time.sleep(20)
                continue
            props = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            receipt = json.loads(args.after_receipt.read_text()) if args.after_receipt.exists() else None
            status["predecessor"] = props
            save(queue, status)
            if predecessor_ready(props, receipt, args.after_invocation):
                break
            time.sleep(20)
        for name, digest in json.loads(args.manifest.read_text()).items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen prefill diagnostic source changed: " + name)
        model = args.source / "models/demos/qwen38_27b_qb2"
        env = environment(args.task, args.source, args.weights)
        for key in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "TT_METAL_DISABLE_SFPLOADMACRO",
            "TT_METAL_KERNEL_PATH",
            "TT_VISIBLE_DEVICES",
        ):
            env.pop(key, None)
        env.update(
            QWEN_PREFILL_DIAGNOSTIC="1",
            QWEN_PREFILL_DIAGNOSTIC_RECEIPT=str(args.output / "diagnostic.json"),
            QWEN_PRECISION_CONFIG=str(model / "config/precision_single_step_compact_gdn_resident_gates_bfp8_all.json"),
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_prefill_attention_diagnostic.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=1620",
            f"--junitxml={args.output}/hardware.xml",
        ]
        status.update(state="hardware_or_waiting_for_lock", hardware_started=True, command=command)
        save(queue, status)
        run_capture(command, cwd=args.source, env=env, root=args.output, timeout=1800)
        validation = validate_report(json.loads((args.output / "diagnostic.json").read_text()))
        suites = ET.parse(args.output / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise ValueError("Prefill numerical diagnostic test was failed or skipped")
        status.update(state="completed", cleanup_completed=True, validation=validation)
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(queue, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "after-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-invocation", required=True)
    run(parser.parse_args())
