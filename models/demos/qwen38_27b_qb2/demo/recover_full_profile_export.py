# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Filtered CPU-only export from a completed, immutable full-model Tracy capture."""

import argparse
import hashlib
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import collect
from models.demos.qwen38_27b_qb2.tests.profile_export_recovery import validate_pair


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def passed_xml(path):
    suites = ET.parse(path).getroot().findall(".//testsuite")
    if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise ValueError("Original hardware test failed, was skipped, or is incomplete")


def run(args):
    profile = json.loads((args.original / "profile.json").read_text())
    baseline = json.loads((args.baseline / "profile.json").read_text())
    validate_pair(profile, baseline)
    for root in (args.original, args.baseline):
        passed_xml(root / "hardware.xml")
    args.output.mkdir()
    status = dict(
        state="exporting",
        physical_devices_accessed=False,
        cleanup_completed=False,
        original=str(args.original),
        baseline=str(args.baseline),
        started_at=time.time(),
        files={},
    )
    queue = args.output / "queue.json"
    save(queue, status)
    try:
        logs = args.output / "tracy/.logs"
        logs.mkdir(parents=True)
        budget = dict(maximum_total=8 * 1024**3, maximum_file=2 * 1024**3, minimum_free=16 * 1024**3)
        if shutil.disk_usage(args.output).free < budget["minimum_free"] + 4 * 1024**3:
            raise ValueError("Insufficient host-disk headroom for bounded export")
        for name in ("profile.json", "hardware.xml"):
            shutil.copy2(args.original / name, args.output / name)
        for name in (
            "tracy_profile_log_host.tracy",
            "cpp_device_perf_report.csv",
            "zone_src_locations.log",
            "new_zone_src_locations.log",
        ):
            source = args.original / "tracy/.logs" / name
            if not source.exists() and name.endswith("zone_src_locations.log"):
                continue
            minimum_size = 0 if name.endswith("zone_src_locations.log") else 1
            if (
                source.is_symlink()
                or not source.is_file()
                or not minimum_size <= source.stat().st_size <= budget["maximum_file"]
            ):
                raise ValueError("Missing or oversized completed capture: " + name)
            expected = digest(source)
            shutil.copy2(source, logs / name)
            if digest(logs / name) != expected:
                raise ValueError("Capture changed during copy")
            status["files"][name] = dict(bytes=source.stat().st_size, sha256=expected)
        save(queue, status)
        for name, flags in (
            ("tracy_ops_times.csv", ["-u", "-f", "TT_", "-t", "TT_"]),
            ("tracy_ops_data.csv", ["-m", "-s", ";"]),
        ):
            # No shell: only this new directory receives writes. Export all
            # messages/signposts, but avoid irrelevant CPU child-zone timing.
            command = [
                str(args.python),
                "-c",
                "import subprocess,sys; f=open(sys.argv[1],'w'); subprocess.run(sys.argv[2:],stdout=f,check=True)",
                str(logs / name),
                str(args.exporter),
                *flags,
                str(logs / "tracy_profile_log_host.tracy"),
            ]
            run_capture(
                command, cwd=args.output, env=dict(os.environ), root=args.output, timeout=900, artifact_budget=budget
            )
            status["files"][name] = dict(bytes=(logs / name).stat().st_size, sha256=digest(logs / name))
            save(queue, status)
        status["state"] = "processing"
        save(queue, status)
        env = dict(os.environ, TT_METAL_PROFILER_DIR=str(args.output / "tracy"))
        run_capture(
            [
                str(args.python),
                "-c",
                "from tracy.process_ops_logs import process_ops; import sys; from pathlib import Path; process_ops(Path(sys.argv[1]),None,True)",
                str(args.output / "tracy"),
            ],
            cwd=args.output,
            env=env,
            root=args.output,
            timeout=900,
            artifact_budget=budget,
        )
        analysis = collect(args.output)
        if analysis["full_trace_reconciliation_passed"] is not True:
            raise ValueError("Recovered export does not reconcile with hardware host time")
        status.update(
            state="completed",
            cleanup_completed=True,
            comparisons=analysis["comparisons"],
            full_trace_reconciliation_passed=True,
            outputs_and_inputs_match=True,
            baseline_replays=baseline["replays"],
            profiled_replays=profile["replays"],
        )
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(queue, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "baseline", "output", "exporter", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
