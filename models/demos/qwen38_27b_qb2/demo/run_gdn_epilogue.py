# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Queue physical epilogue qualification after a clean BFP8 comparison run."""

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
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import BATCHES, PLACEMENTS, compare_timings, predecessor_ready


def run(args):
    args.output.mkdir()
    status_path = args.output / "queue.json"
    status = dict(
        state="waiting",
        hardware_started=False,
        cleanup_completed=False,
        predecessor_unit=args.after_unit,
        predecessor_invocation=args.after_invocation,
        hardware_lock="/tmp/tt-device.lock",
        survives_disconnect=True,
        resumes_after_reboot=False,
        promoted_to_model=False,
        started_at=time.time(),
    )
    manifest = json.loads(args.manifest.read_text())

    def verify_source():
        for relative, digest in manifest.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen source changed: " + relative)

    def terminate(signum, frame):
        raise InterruptedError(f"Epilogue experiment received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(status_path, status)
    try:
        verify_source()
        deadline = time.monotonic() + args.wait_timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Predecessor wait expired; no hardware opened")
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
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                status["observation_error"] = type(error).__name__
                save(status_path, status)
                time.sleep(20)
                continue
            properties = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            receipt = json.loads(args.after_receipt.read_text()) if args.after_receipt.exists() else None
            status["predecessor"] = properties
            save(status_path, status)
            if predecessor_ready(properties, receipt, args.after_invocation):
                break
            time.sleep(20)
        verify_source()
        model = args.source / "models/demos/qwen38_27b_qb2"
        env = environment(args.task, args.source, args.weights)
        for key in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_DISABLE_SFPLOADMACRO",
            "TT_METAL_KERNEL_PATH",
            "TT_VISIBLE_DEVICES",
        ):
            env.pop(key, None)
        env.update(
            QWEN_GDN_EPILOGUE="1",
            QWEN_GDN_EPILOGUE_RECEIPT=str(args.output / "epilogue.json"),
            QWEN_PRECISION_CONFIG=str(model / "config/precision_accurate_decode_bfp8_all.json"),
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_gdn_epilogue.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=3600",
            f"--junitxml={args.output}/hardware.xml",
        ]
        status.update(state="hardware_or_waiting_for_lock", hardware_started=True, command=command)
        save(status_path, status)
        run_capture(command, cwd=args.source, env=env, root=args.output, timeout=5400)
        report = json.loads((args.output / "epilogue.json").read_text())
        if (
            report.get("state") != "completed"
            or report.get("passed") is not True
            or report.get("cleanup_completed") is not True
        ):
            raise ValueError("Epilogue hardware test lacks successful completion and cleanup")
        if [(row["placement"], row["batch"]) for row in report["cases"]] != [
            (placement, batch) for placement in PLACEMENTS for batch in BATCHES
        ]:
            raise ValueError("Incomplete epilogue placement/batch coverage")
        for row in report["cases"]:
            if not row["passed"] or compare_timings(row["timings"]) != row["comparison"]:
                raise ValueError("Incorrect epilogue timing summary")
        suites = ET.parse(args.output / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise ValueError("Physical epilogue test failed or was skipped")
        status.update(
            state="completed",
            cleanup_completed=True,
            all_timing_comparisons_qualified=all(
                row["comparison"]["timing_comparison_qualified"] for row in report["cases"]
            ),
        )
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "after-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-invocation", required=True)
    parser.add_argument("--wait-timeout", type=int, default=115200)
    run(parser.parse_args())
