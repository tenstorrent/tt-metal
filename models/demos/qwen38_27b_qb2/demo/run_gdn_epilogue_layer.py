# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight epilogue integration test under the shared device lock."""

import argparse
import hashlib
import json
import signal
import time
import xml.etree.ElementTree as ET

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import BATCHES, compare


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
        raise InterruptedError(f"Epilogue integration received signal {signum}")

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
            QWEN_GDN_EPILOGUE_LAYER="1",
            QWEN_GDN_LAYER_RECEIPT=str(args.output / "layer.json"),
            QWEN_PRECISION_CONFIG=str(model / "config/precision_single_step_shared_qk_epilogue_bfp8_all.json"),
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_gdn_epilogue_layer.py"),
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
        report = json.loads((args.output / "layer.json").read_text())
        if (
            report.get("state") != "completed"
            or report.get("passed") is not True
            or report.get("cleanup_completed") is not True
        ):
            raise ValueError("Real-weight test lacks passing result and clean shutdown")
        if len(report["cases"]) != 3 * len(BATCHES) or [row["batch"] for row in report["comparisons"]] != list(BATCHES):
            raise ValueError("Incomplete real-weight batch coverage")
        for index, summary in enumerate(report["comparisons"]):
            if compare(report["cases"][index * 3 : index * 3 + 3]) != summary:
                raise ValueError("Real-weight comparison does not match raw cases")
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
