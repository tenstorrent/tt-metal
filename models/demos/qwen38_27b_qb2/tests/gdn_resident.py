# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only gate for the isolated register-resident recurrence experiment."""

import math

from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings

CASES = ((16, 1), (32, 1), (1, 64))


def checkpoints(cycles):
    return [("cancellation", 1)] + [
        ("trace", (cycle + 1) * 64)
        for cycle in range(cycles)
        if cycle == 0 or (cycle + 1) % 4 == 0 or cycle + 1 == cycles
    ]


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def dense_pass(metrics):
    names = ("min_head_pcc", "max_head_relative_rms", "max_abs")
    if metrics.get("passed") is not True or metrics.get("finite") is not True:
        return False
    if any(type(metrics.get(k)) not in (int, float) or not math.isfinite(metrics[k]) for k in names):
        return False
    return (
        0.999 <= metrics["min_head_pcc"] <= 1.000001
        and 0 <= metrics["max_head_relative_rms"] <= 0.005
        and metrics["max_abs"] >= 0
    )


def validate_report(report):
    if any(report.get(k) is not True for k in ("passed", "cleanup_completed")) or report.get("state") != "completed":
        raise ValueError("Resident recurrence lacks complete clean evidence")
    devices = report.get("device_ids", [])
    if len(devices) != 4 or any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != 4:
        raise ValueError("Require all four distinct physical TP ranks")
    cases = report.get("cases", [])
    if [(case.get("batch"), case.get("cycles")) for case in cases] != list(CASES):
        raise ValueError("Missing B16/B32 or 4K long-horizon case")
    comparisons = []
    for case in cases:
        if (
            case.get("passed") is not True
            or case.get("input_and_address_stability") is not True
            or case.get("heads_per_rank") != case["batch"] * 12
            or case.get("steps") != case["cycles"] * 64
        ):
            raise ValueError("Incomplete recurrence trajectory or buffer lifetime check")
        if [(c.get("phase"), c.get("steps")) for c in case.get("checks", [])] != checkpoints(case["cycles"]):
            raise ValueError("Missing cancellation or intermediate trajectory checks")
        for check in case["checks"]:
            ranks = check.get("ranks", [])
            if [r.get("rank") for r in ranks] != list(range(4)):
                raise ValueError("State/output checks must cover all TP ranks")
            for rank in ranks:
                if (
                    rank.get("state_bit_identical") is not True
                    or rank.get("output_bit_identical") is not True
                    or not dense_pass(rank.get("state_dense", {}))
                    or not dense_pass(rank.get("output_dense", {}))
                    or not valid_hash(rank.get("state_sha256"))
                    or not valid_hash(rank.get("output_sha256"))
                ):
                    raise ValueError("Control identity or per-head dense accuracy failed")
        if case["batch"] in (16, 32):
            if case.get("identical_steps_per_arm") != 508:
                raise ValueError("Timing arms do not have matching trajectories")
            for field in ("timing_final_states", "timing_final_outputs"):
                values = case.get(field, [])
                if (
                    len(values) != 3
                    or any(len(v) != 4 or not all(valid_hash(h) for h in v) for v in values)
                    or not values[0] == values[1] == values[2]
                ):
                    raise ValueError("Timing arms disagree on all-rank final state/output")
            comparison = compare_timings(case.get("timings", []))
            if comparison != case.get("comparison"):
                raise ValueError("Timing summary disagrees with raw samples")
            # One read/write of the FP32 state at modeled 512 GB/s per chip.
            # This omits inputs/output and is a traffic floor, not a counter.
            bound_us = case["heads_per_rank"] * 128 * 128 * 4 * 2 / (512 * 1000)
            comparisons.append(
                dict(
                    batch=case["batch"],
                    state_only_bandwidth_floor_us=bound_us,
                    p1_target_us=bound_us + 10,
                    p1_target_met=comparison["timing_comparison_qualified"] and comparison["fused_us"] <= bound_us + 10,
                    **comparison,
                )
            )
    return dict(correctness_passed=True, full_model_qualified=False, promoted_to_serving=False, comparisons=comparisons)
