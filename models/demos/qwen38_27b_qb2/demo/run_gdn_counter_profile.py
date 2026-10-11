# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded independent counter passes with exact output checks and retained raw logs."""

import argparse
import hashlib
import json
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_gdn_phase_profile import run as phase_run
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.bounded_profile import check_artifact_budget
from models.demos.qwen38_27b_qb2.tests.gdn_counter_profile import analyze_pass, compare_outputs, counter_names


def run(args):
    # Import the installed planner only at runtime; do not copy its masks or
    # open a device to detect an architecture already pinned by the allocation.
    from tracy import perf_counter_analysis, perf_counter_multipass

    groups = perf_counter_multipass.resolve_perf_counter_groups(args.groups.split(","), "blackhole")
    if not groups or (args.groups != "all" and set(groups) != set(args.groups.split(","))):
        raise ValueError("Counter request must resolve exactly")
    # The installed Galaxy firmware overflows by eight bytes with L1_0+FPU,
    # despite the native planner's general three-group ceiling. Keep its mux
    # planning, using the stricter measured limit for this allocation.
    passes = perf_counter_multipass.schedule_perf_counter_passes(groups, max_groups_per_pass=1)
    passes.sort(key=lambda requested: min(groups.index(group) for group in requested))
    header = args.task / "metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/hw_counters.h"
    args.output.mkdir()
    queue = args.output / "queue.json"
    status = dict(
        state="preflight",
        groups=groups,
        native_passes=passes,
        max_groups_per_pass=1,
        runs=[],
        hardware_lock="/tmp/tt-device.lock",
        survives_disconnect=True,
        resumes_after_reboot=False,
        cleanup_completed=False,
        precision_change=False,
        promoted_to_serving=False,
        physical_dram_utilization_measured=False,
    )
    native = [header, Path(perf_counter_multipass.__file__), Path(perf_counter_analysis.__file__)]
    status["native_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in native}
    save(queue, status)
    baseline = None
    try:
        if args.reset_first:
            status["state"] = "locked_recovery"
            save(queue, status)
            run_capture(
                ["flock", "--exclusive", "--timeout", "60", "/tmp/tt-device.lock", "tt-smi", "-glx_reset"],
                cwd=args.source,
                env=environment(args.task, args.source, args.weights),
                root=args.output,
                timeout=300,
            )
            status["locked_reset_completed"] = True
        plan = [("before", [])] + [(f"pass-{i:02d}", p) for i, p in enumerate(passes)] + [("after", [])]
        for label, requested in plan:
            check_artifact_budget(args.output, maximum_total=16 * 1024**3)
            for path, digest in status["native_sha256"].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise ValueError("Native counter source changed during capture")
            reused = label == "before" and args.baseline is not None
            output = args.baseline if reused else args.output / label
            row = dict(label=label, groups=requested, state="running_or_waiting_for_lock", output=str(output))
            status["runs"].append(row)
            status["state"] = row["state"]
            save(queue, status)
            if reused:
                previous = json.loads((output / "queue.json").read_text())
                if previous.get("state") != "completed" or previous.get("cleanup_completed") is not True:
                    raise ValueError("Reused control requires successful capture and clean teardown")
                row["reused_completed_control"] = True
            else:
                phase_run(
                    argparse.Namespace(
                        task=args.task,
                        source=args.source,
                        weights=args.weights,
                        output=output,
                        manifest=args.manifest,
                        pipeline=True,
                        after_unit=None,
                        after_invocation=None,
                        after_receipt=None,
                        counter_groups=",".join(requested) if requested else None,
                    )
                )
            receipt = json.loads((output / "phase.json").read_text())
            expected_mask = perf_counter_multipass.perf_counter_groups_to_bitfield(requested)
            if receipt.get("counter_mask") != expected_mask:
                raise ValueError("Counter receipt does not match requested native pass")
            if baseline is None:
                baseline = receipt
            compare_outputs(baseline, receipt)
            expected = counter_names(header.read_text(), requested)
            analysis = analyze_pass(output, expected, perf_counter_analysis.COUNTER_TYPE_NAMES)
            save(args.output / "reused-control-analysis.json" if reused else output / "counters.json", analysis)
            row.update(state="completed", exact_output_match=True, counter_mask=receipt["counter_mask"])
            save(queue, status)
        check_artifact_budget(args.output, maximum_total=16 * 1024**3)
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
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--baseline", type=Path, help="Reuse a completed mask-zero control after analysis-only failure")
    parser.add_argument("--reset-first", action="store_true", help="Locked recovery after an owned unclean failure")
    run(parser.parse_args())
