# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded GDN reader, compute and writer timing zones under the shared device lock."""

import argparse
import hashlib
import json
import signal
import time
import xml.etree.ElementTree as ET

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save


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
        raise InterruptedError(f"GDN phase profiler received signal {signum}")

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
            "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES",
            "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
        ):
            env.pop(key, None)
        env.update(
            QWEN_GDN_PHASE_PROFILE="1",
            QWEN_GDN_PHASE_RECEIPT=str(args.output / "phase.json"),
            TT_METAL_PROFILER_DIR=str(args.output / "tracy"),
            TRACY_NO_WEB_SERVER="1",
        )
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            "--profile-ops",
            str(model / "tests/test_gdn_phase_profile.py"),
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
        report = json.loads((args.output / "phase.json").read_text())
        if (
            report.get("state") != "completed"
            or report.get("passed") is not True
            or report.get("cleanup_completed") is not True
        ):
            raise ValueError("Phase profile test lacks passing result and clean shutdown")
        expected = [(b, n, profiled) for b in (32, 16) for n in (1, 2) for profiled in (False, True)]
        if [(r["batch"], r["buffers"], r["profiled"]) for r in report["cases"]] != expected or report[
            "kernel_calls"
        ] != 24:
            raise ValueError("Incomplete phase profile coverage")
        raw_files = list((args.output / "tracy").rglob("profile_log_device.csv"))
        if not raw_files or not all(p.stat().st_size for p in raw_files):
            raise ValueError("Missing raw device zones; do not infer NoC or CB wait attribution")
        markers = set()
        required = {
            "GDN_READER_CB_RESERVE",
            "GDN_READER_DRAM_ISSUE_AND_WAIT",
            "GDN_READER_L1_PREPARE",
            "GDN_COMPUTE_INPUT_WAIT",
            "GDN_COMPUTE_DELTA",
            "GDN_COMPUTE_STATE_UPDATE",
            "GDN_COMPUTE_OUTPUT",
            "GDN_WRITER_STATE_CB_WAIT",
            "GDN_WRITER_DRAM_ISSUE_AND_OUTPUT_WAIT",
            "GDN_WRITER_OUTPUT_AND_BARRIER",
        }
        for path in raw_files:
            with path.open() as stream:
                for line in stream:
                    markers.update(label for label in required if label in line)
        if markers != required:
            raise ValueError("Missing required phase markers: " + str(sorted(required - markers)))
        status["raw_device_logs"] = [str(p) for p in raw_files]
        status["required_markers_present"] = sorted(markers)
        status["causal_bottleneck_proven"] = False
        suites = ET.parse(args.output / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise ValueError("Phase profile hardware test failed or was skipped")
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
