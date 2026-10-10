# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coverage and timing gates for skipping unused epilogue input rows."""

from models.demos.qwen38_27b_qb2.tests.compact_gdn import changing_input_checkpoints
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings

MODES = ((False, True, 0), (True, True, 0), (True, True, 2560))
CASES = [(b, p, m) for b in (16, 32) for p in ("l1", "dram") for m in MODES]
CASES += [(b, p, MODES[-1]) for b in (1, 17, 31) for p in ("l1", "dram")]
LAYERS = [(b, p) for b in (16, 32) for p in ("skip", "poison")]


def rank_hashes(values):
    return (
        isinstance(values, list)
        and len(values) == 4
        and all(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in values)
    )


def validate_report(report):
    if report.get("state") != "completed" or report.get("cleanup_completed") is not True:
        raise ValueError("Epilogue padding experiment did not complete cleanly")
    devices = report.get("device_ids", [])
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("Require four distinct physical ranks")
    cases = report.get("cases", [])
    if len(cases) != len(CASES) or {
        (c.get("batch"), c.get("placement"), tuple(c.get("mode", []))) for c in cases
    } != set(CASES):
        raise ValueError("Incomplete batch, placement or gate layout coverage")
    timings = []
    for case in cases:
        if not all(
            case.get(k) is True
            for k in (
                "passed",
                "padding_experiment",
                "input_and_address_stability",
                "changed_input_trace",
                "output_padding_zero",
            )
        ):
            raise ValueError("Padding, replay or input ownership checks are incomplete")
        if case.get("input_cb_slots") != 2 or (case["batch"] == 32 and case.get("max_items_per_core", 0) <= 2):
            raise ValueError("B32 must exercise both CB slots and wrap to a reused slot")
        checks = case.get("checks", [])
        if [c.get("allocation") for c in checks] != [0, 1, 0] or checks != case.get("poison_checks"):
            raise ValueError("Require identical skip and poisoned A/B/A outputs")
        for field in ("norm_sha256", "output_sha256"):
            if any(not rank_hashes(c.get(field)) for c in checks):
                raise ValueError("Missing exact normalization or output comparison on every rank")
            if checks[0][field] != checks[2][field] or checks[0][field] == checks[1][field]:
                raise ValueError("Changed inputs were ignored or not restored")
        replay = case.get("replay_checks", [])
        if [r.get("input_padding") for r in replay] != ["skip", "poison"] or any(
            r.get("output_sha256") != [c["output_sha256"] for c in checks] for r in replay
        ):
            raise ValueError("Both captured programs must replay changed inputs exactly")
        if case.get("timing_input_padding") != ["zero", "skip", "zero"]:
            raise ValueError("Time only the zero/skip/zero bracket; poisoning is not a speed candidate")
        measured = compare_timings(case["timings"])
        if measured != case.get("comparison"):
            raise ValueError("Timing comparison differs from retained samples")
        timings.append(dict(batch=case["batch"], placement=case["placement"], mode=case["mode"], **measured))
    layers = report.get("layers", [])
    if len(layers) != len(LAYERS) or {(c.get("batch"), c.get("input_padding")) for c in layers} != set(LAYERS):
        raise ValueError("Require B16/B32 real-weight checks for skipped and poisoned padding")
    for layer in layers:
        if (
            layer.get("updates") != 4096
            or layer.get("all_ranks_bit_identical") is not True
            or layer.get("persistent_sessions_precede_trace_capture") is not True
        ):
            raise ValueError("Require independent persistent sessions and 4096 exact layer updates")
        checks = layer.get("checkpoints", [])
        if tuple(c.get("step") for c in checks) != changing_input_checkpoints(4096):
            raise ValueError("Missing layer checkpoints")
        for check in checks:
            hashes = check.get("recurrent_conv_projected_sha256", [])
            if (
                check.get("all_values_finite") is not True
                or len(hashes) != 3
                or any(not rank_hashes(ranks) for ranks in hashes)
            ):
                raise ValueError("Missing finite state, history or projected output evidence")
    return dict(correctness_passed=True, timings=timings, full_model_qualified=False, promoted_to_serving=False)
