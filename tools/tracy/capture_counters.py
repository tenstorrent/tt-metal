# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
One-command, repeatable perf-counter capture (Phase 5 of the Nsight plan).

Wraps ``python -m tracy`` with the repeatability levers the plan requires:
  * a buffer size pinned to cover the workload (derived from the counter group
    set + ops/device, not guessed) and held constant across runs so the kernel
    hash stays stable (100% JIT cache hit);
  * optional deterministic compute-core sampling;
  * artifacts (CSV + .tracy + the exact env/scope) archived to a durable dir,
    because the reports dir is wiped each run.

``build_capture_plan`` is the pure, testable core; ``main`` runs it on device
and archives the result.
"""

import argparse
import json
import os
import shutil

from tracy.perf_counter_sizing import (
    bytes_per_program,
    counters_per_zone,
    marker_slots_per_zone,
    normalized_counter_groups,
    recommend_program_support_count,
    single_pass_l1_headroom,
)
from tracy.perf_counter_multipass import schedule_perf_counter_passes


def build_capture_plan(
    groups,
    programs_per_device,
    arch,
    compute_core_sample=None,
    archive_root=None,
    timestamp=None,
):
    """Compose a deterministic capture plan: pinned buffer, env, warnings.

    Returns a dict with: groups, program_support_count, l1_headroom, warnings,
    env (vars to apply to the run), and archive_dir.
    """
    groups = normalized_counter_groups(arch, groups)
    if not groups:
        raise ValueError("at least one counter group is required")
    passes = []
    for pass_groups in schedule_perf_counter_passes(groups):
        passes.append(
            {
                "groups": pass_groups,
                "counter_records_per_zone": counters_per_zone(arch, pass_groups),
                "counter_marker_slots_per_zone": marker_slots_per_zone(arch, pass_groups),
                "estimated_bytes_per_program_per_risc": bytes_per_program(arch, pass_groups),
                "program_support_count": recommend_program_support_count(
                    programs_per_device, arch=arch, groups=pass_groups
                ),
                "l1_headroom": single_pass_l1_headroom(arch, pass_groups),
            }
        )
    program_support_count = max(p["program_support_count"] for p in passes)
    headroom = min(p["l1_headroom"] for p in passes)

    warnings = []
    if headroom < 0:
        warnings.append(
            f"a scheduled pass exceeds the L1 optional-marker budget by {-headroom} slots; "
            "BRISC must flush during readout; verify DRAM capacity and dropped-marker flags"
        )

    env = {
        "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT": str(program_support_count),
        "TT_METAL_DEVICE_ARCH": arch,
    }
    if compute_core_sample:
        env["TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE"] = str(compute_core_sample)

    archive_dir = None
    if archive_root and timestamp:
        archive_dir = os.path.join(archive_root, timestamp)

    return {
        "groups": list(groups),
        "passes": passes,
        "programs_between_drains": programs_per_device,
        "optional_marker_reserve": 16,
        "safety": 1.2,
        "program_support_count": program_support_count,
        "l1_headroom": headroom,
        "warnings": warnings,
        "env": env,
        "archive_dir": archive_dir,
    }


def _archive_artifacts(report_csv, archive_dir, plan):
    os.makedirs(archive_dir, exist_ok=True)
    report_dir = os.path.dirname(report_csv)
    for fname in os.listdir(report_dir):
        if fname.endswith(".csv") or fname.endswith(".tracy"):
            shutil.copy2(os.path.join(report_dir, fname), archive_dir)
    with open(os.path.join(archive_dir, "capture_scope.txt"), "w") as f:
        f.write(f"groups: {','.join(plan['groups'])}\n")
        f.write(f"program_support_count: {plan['program_support_count']}\n")
        for k, v in plan["env"].items():
            f.write(f"{k}={v}\n")
    with open(os.path.join(archive_dir, "capture_plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="repeatable perf-counter capture")
    parser.add_argument("--test", required=True, help="pytest target to profile")
    parser.add_argument("--groups", required=True, help="comma-separated counter groups (e.g. fpu,instrn)")
    parser.add_argument(
        "--programs-per-device", type=int, required=True, help="maximum program invocations per device between drains"
    )
    parser.add_argument("--arch", default=os.environ.get("ARCH_NAME", "blackhole"))
    parser.add_argument("--compute-core-sample", type=int, default=None)
    parser.add_argument("--name", default="CounterCapture")
    parser.add_argument("--archive-root", default=os.path.expanduser("~/traces"))
    args = parser.parse_args(argv)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    plan = build_capture_plan(
        groups,
        args.programs_per_device,
        args.arch,
        compute_core_sample=args.compute_core_sample,
        archive_root=args.archive_root,
        timestamp=timestamp,
    )

    for w in plan["warnings"]:
        print(f"WARNING: {w}")

    for k, v in plan["env"].items():
        os.environ[k] = v

    from tracy.process_model_log import run_device_profiler, get_latest_ops_log_filename

    run_device_profiler(
        f"pytest {args.test}",
        args.name,
        capture_perf_counters_groups=plan["groups"],
        op_support_count=plan["program_support_count"],
    )

    report_csv = str(get_latest_ops_log_filename(args.name))
    _archive_artifacts(report_csv, plan["archive_dir"], plan)
    print(f"archived capture to {plan['archive_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
