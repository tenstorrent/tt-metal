# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded GDN reader, compute and writer timing zones under the shared device lock."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.demo.watch_profile_export import properties
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import predecessor_ready
from models.demos.qwen38_27b_qb2.tests.gdn_phase_profile import required_markers, validate_coverage


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
        pipeline=args.pipeline,
        predecessor_unit=args.after_unit,
        predecessor_invocation=args.after_invocation,
        started_at=time.time(),
    )

    def terminate(signum, frame):
        raise InterruptedError(f"GDN phase profiler received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(queue, status)
    try:
        if args.after_unit:
            status["state"] = "waiting"
            deadline = time.monotonic() + 24 * 3600
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("Predecessor remains live; no restart or hardware takeover")
                try:
                    props = properties(args.after_unit)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    status["observation_error"] = type(error).__name__
                    save(queue, status)
                    time.sleep(20)
                    continue
                receipt = json.loads(args.after_receipt.read_text()) if args.after_receipt.exists() else None
                status["predecessor"] = props
                save(queue, status)
                if predecessor_ready(props, receipt, args.after_invocation):
                    break
                time.sleep(20)
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
            "TT_METAL_PROFILE_PERF_COUNTERS",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "QWEN_GDN_PHASE_PROFILE",
            "QWEN_GDN_PIPELINE_PROFILE",
        ):
            env.pop(key, None)
        env.update(
            QWEN_GDN_PHASE_RECEIPT=str(args.output / "phase.json"),
            TT_METAL_PROFILER_DIR=str(args.output / "tracy"),
            TRACY_NO_WEB_SERVER="1",
        )
        env["QWEN_GDN_PIPELINE_PROFILE" if args.pipeline else "QWEN_GDN_PHASE_PROFILE"] = "1"
        groups = getattr(args, "counter_groups", None)
        if groups:
            from tracy.perf_counter_multipass import (
                perf_counter_groups_to_bitfield,
                resolve_perf_counter_groups,
                schedule_perf_counter_passes,
            )

            requested = groups.split(",")
            resolved = resolve_perf_counter_groups(requested, "blackhole")
            if set(requested) != set(resolved) or len(schedule_perf_counter_passes(resolved)) != 1:
                raise ValueError("Counter capture requires one exact native-planned pass")
            mask = perf_counter_groups_to_bitfield(resolved)
            env["TT_METAL_PROFILE_PERF_COUNTERS"] = str(mask)
            status.update(counter_groups=resolved, counter_mask=mask)
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            "--profile-counters" if groups else "--profile-ops",
            str(model / ("tests/test_gdn_pipeline_profile.py" if args.pipeline else "tests/test_gdn_phase_profile.py")),
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
        validate_coverage(report, pipeline=args.pipeline)
        if report.get("counter_mask", 0) != status.get("counter_mask", 0):
            raise ValueError("Hardware receipt counter mask differs from requested pass")
        raw_files = list((args.output / "tracy").rglob("profile_log_device.csv"))
        if not raw_files or not all(p.stat().st_size for p in raw_files):
            raise ValueError("Missing raw device zones; do not infer NoC or CB wait attribution")
        markers = set()
        required = required_markers(pipeline=args.pipeline)
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
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--counter-groups", help="One native-planned comma-separated counter pass")
    parser.add_argument("--after-unit")
    parser.add_argument("--after-invocation")
    parser.add_argument("--after-receipt", type=Path)
    args = parser.parse_args()
    predecessor = (args.after_unit, args.after_invocation, args.after_receipt)
    if any(predecessor) and not all(predecessor):
        parser.error("All three predecessor arguments are required together")
    run(args)
