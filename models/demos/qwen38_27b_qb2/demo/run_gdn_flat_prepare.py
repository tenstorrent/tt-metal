# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-input preparation test under the shared device lock."""

import argparse
import hashlib
import json
import signal
import time
import xml.etree.ElementTree as ET

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings


def run(args):
    args.output.mkdir()
    queue = args.output / "queue.json"
    status = dict(
        state="preflight",
        cleanup_completed=False,
        hardware_lock="/tmp/tt-device.lock",
        survives_disconnect=True,
        resumes_after_reboot=False,
        promoted_to_serving=False,
        started_at=time.time(),
    )

    def terminate(signum, frame):
        raise InterruptedError(f"Direct preparation received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(queue, status)
    try:
        manifest = json.loads(args.manifest.read_text())
        for relative, digest in manifest.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen source changed: " + relative)
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
            QWEN_GDN_FLAT_PREPARE="1",
            QWEN_GDN_FLAT_PREPARE_RECEIPT=str(args.output / "prepare.json"),
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_gdn_flat_prepare.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=1200",
            f"--junitxml={args.output}/hardware.xml",
        ]
        status.update(state="hardware_or_waiting_for_lock", command=command)
        save(queue, status)
        # The outer deadline includes lock waiting; pytest's independent bound
        # starts only after the safe runner obtains exclusive hardware access.
        run_capture(command, cwd=args.source, env=env, root=args.output, timeout=5400)
        report = json.loads((args.output / "prepare.json").read_text())
        if (
            report.get("state") != "completed"
            or report.get("passed") is not True
            or report.get("cleanup_completed") is not True
        ):
            raise ValueError("Preparation test lacks passing result and clean shutdown")
        expected = [(b, t, p) for b in (32, 16) for t in (1, 32) for p in ("l1", "dram")] + [(1, 32, "dram")]
        actual = [(case["batch"], case["time_rows"], case["placement"]) for case in report["cases"]]
        if actual != expected or not all(case["passed"] for case in report["cases"]):
            raise ValueError("Incomplete preparation coverage")
        for case in report["cases"]:
            if compare_timings(case["timings"]) != case["comparison"]:
                raise ValueError("Preparation comparison does not match raw cases")
        suites = ET.parse(args.output / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise ValueError("Hardware integration test failed or was skipped")
        status.update(state="completed", cleanup_completed=True)
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(queue, status)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
