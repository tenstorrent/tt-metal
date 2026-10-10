# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coverage and timing gates for packed user-row GDN gates."""

from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings

CASES = [(b, p) for b in (1, 16, 17, 31, 32) for p in ("l1", "dram")]


def validate_report(report):
    if report.get("state") != "completed" or report.get("cleanup_completed") is not True:
        raise ValueError("Compact gate screen did not complete cleanly")
    devices = report.get("device_ids", [])
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("Require four distinct device ranks")
    cases = report.get("cases", [])
    if len(cases) != len(CASES) or {(c.get("batch"), c.get("placement")) for c in cases} != set(CASES):
        raise ValueError("Missing compact gate shape/placement coverage")
    timings = []
    for case in cases:
        if not all(case.get(k) is True for k in ("input_immutability", "stable_addresses", "changed_input_trace")):
            raise ValueError("Compact gates lack input/replay validation")
        checks = case.get("checks", [])
        if [c.get("allocation") for c in checks] != [0, 1, 0]:
            raise ValueError("Require A/B/A allocation coverage")
        for check in checks:
            hashes = check.get("prepared_sha256", [])
            if check.get("bit_identical") is not True or check.get("finite") is not True or len(hashes) != 4:
                raise ValueError("Missing exact prepared operand comparison")
            for ranks in hashes:
                if len(ranks) != 4 or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in ranks):
                    raise ValueError("Missing prepared operand hashes on all ranks")
        if (
            checks[0]["prepared_sha256"] != checks[2]["prepared_sha256"]
            or checks[0]["prepared_sha256"] == checks[1]["prepared_sha256"]
        ):
            raise ValueError("Changed inputs were not restored or were ignored")
        measured = compare_timings(case["timings"])
        if measured != case.get("comparison"):
            raise ValueError("Stored compact-gate timing differs from samples")
        timings.append(dict(batch=case["batch"], placement=case["placement"], **measured))
    layers = report.get("layers", [])
    if len(layers) != 2 or {c.get("batch") for c in layers} != {16, 32}:
        raise ValueError("Missing real-weight B16/B32 coverage")
    for layer in layers:
        checks = layer.get("checkpoints", [])
        if [c.get("step") for c in checks] != [1, 2, 4, 8, 16, 32, 64]:
            raise ValueError("Missing changing-input real-weight checks")
        for check in checks:
            hashes = check.get("state_history_output_sha256", [])
            if check.get("bit_identical") is not True or check.get("finite") is not True or len(hashes) != 3:
                raise ValueError("Real-weight output/state comparison failed")
            if any(
                len(ranks) != 4 or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in ranks)
                for ranks in hashes
            ):
                raise ValueError("Real-weight check lacks all rank hashes")
    return dict(correctness_passed=True, timings=timings, full_model_qualified=False, promoted_to_serving=False)
