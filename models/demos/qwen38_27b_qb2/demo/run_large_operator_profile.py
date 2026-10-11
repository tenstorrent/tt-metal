# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent bounded native projection/SDPA counters, with mask-zero controls."""

import argparse
import hashlib
import json
import signal
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.bounded_profile import check_artifact_budget
from models.demos.qwen38_27b_qb2.tests.gdn_counter_profile import counter_names
from models.demos.qwen38_27b_qb2.tests.large_operator_profile import (
    analyze_pass,
    capture_plan,
    compare_outputs,
    validate_receipt,
)


def run(args):
    from tracy import perf_counter_analysis, perf_counter_multipass

    requested = args.groups.split(",")
    groups = perf_counter_multipass.resolve_perf_counter_groups(requested, "blackhole")
    if not groups or set(groups) != set(requested):
        raise ValueError("Require exact native counter groups")
    passes = perf_counter_multipass.schedule_perf_counter_passes(groups, max_groups_per_pass=1)
    passes.sort(key=lambda p: min(groups.index(g) for g in p))
    args.output.mkdir()
    queue = args.output / "queue.json"
    status = dict(
        state="preflight",
        runs=[],
        groups=groups,
        passes=passes,
        cleanup_completed=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        hardware_lock="/tmp/tt-device.lock",
        full_model_qualified=False,
        promoted_to_serving=False,
        physical_dram_utilization_measured=False,
    )
    header = args.task / "metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/hw_counters.h"
    frozen = json.loads(args.manifest.read_text())
    native = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (header, Path(perf_counter_multipass.__file__), Path(perf_counter_analysis.__file__))
    }
    status["native_sha256"] = native

    def interrupted(signum, frame):
        raise InterruptedError(f"Operator profiler received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save(queue, status)
    baseline = None
    try:
        plan = capture_plan(passes, args.split_boundaries)
        status["capture_plan"] = plan
        for label, selected, boundary in plan:
            check_artifact_budget(args.output, maximum_total=16 * 1024**3)
            for name, digest in frozen.items():
                if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Frozen source changed: " + name)
            for name, digest in native.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Native counter source changed: " + name)
            output = args.output / label
            output.mkdir()
            env = environment(args.task, args.source, args.weights)
            for key in (
                "TT_METAL_SIMULATOR",
                "TT_METAL_KERNEL_PATH",
                "TT_METAL_DISABLE_SFPLOADMACRO",
                "TT_VISIBLE_DEVICES",
                "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES",
                "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
                "TT_METAL_PROFILE_PERF_COUNTERS",
            ):
                env.pop(key, None)
            mask = perf_counter_multipass.perf_counter_groups_to_bitfield(selected)
            env.update(
                QWEN_LARGE_OPERATOR_PROFILE="1",
                QWEN_LARGE_OPERATOR_RECEIPT=str(output / "operators.json"),
                QWEN_LARGE_OPERATOR_BOUNDARY=boundary,
                TT_METAL_PROFILER_DIR=str(output / "tracy"),
                TRACY_NO_WEB_SERVER="1",
                TT_METAL_PROFILE_PERF_COUNTERS=str(mask),
            )
            command = [
                "/bin/bash",
                str(args.source / "scripts/run_safe_pytest.sh"),
                "--profile-counters",
                str(args.source / "models/demos/qwen38_27b_qb2/tests/test_large_operator_profile.py"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-vv",
                "-s",
                "--tb=short",
                "--timeout=1200",
                f"--junitxml={output}/hardware.xml",
            ]
            row = dict(
                label=label,
                groups=selected,
                boundary_selection=boundary,
                counter_mask=mask,
                state="running_or_waiting_for_lock",
                output=str(output),
                command=command,
            )
            status["runs"].append(row)
            status["state"] = row["state"]
            save(queue, status)
            run_capture(command, cwd=args.source, env=env, root=output, timeout=5400)
            receipt = json.loads((output / "operators.json").read_text())
            validate_receipt(receipt)
            if receipt["counter_mask"] != mask or receipt["boundary_selection"] != boundary:
                raise ValueError("Counter mask or boundary differs from requested pass")
            suites = ET.parse(output / "hardware.xml").getroot().findall(".//testsuite")
            if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ):
                raise ValueError("Hardware test did not pass exactly once")
            status["state"] = "analyzing"
            row["hardware_passed"] = True
            row["cleanup_completed"] = True
            save(queue, status)
            if baseline is None:
                baseline = receipt
            compare_outputs(baseline, receipt)
            analysis = analyze_pass(
                output, counter_names(header.read_text(), selected), perf_counter_analysis.COUNTER_TYPE_NAMES
            )
            save(output / "counters.json", analysis)
            row.update(state="completed", exact_output_match=True, coverage_passed=True)
            save(queue, status)
        status.update(state="completed", cleanup_completed=True)
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        if status["runs"] and status["runs"][-1]["state"] != "completed":
            status["runs"][-1].update(state="failed", detail=str(error)[:1000])
        raise
    finally:
        save(queue, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--groups", default="fpu,pack,unpack,instrn,l1_0,l1_1")
    parser.add_argument(
        "--split-boundaries",
        action="store_true",
        help="One target boundary per counter process avoids profiler-buffer overflow",
    )
    run(parser.parse_args())
