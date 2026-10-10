# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.qwen38_27b_qb2.tests.compact_gdn import (
    BASELINE,
    CANDIDATE,
    COMBINED_GDN_POLICY,
    changing_input_checkpoints,
    validate_long_horizon,
)


def report():
    return dict(
        state="completed",
        cleanup_completed=True,
        baseline=BASELINE,
        candidate=CANDIDATE,
        device_ids=[0, 1, 2, 3],
        cases=[
            dict(
                batch=batch,
                updates=4096,
                all_ranks_bit_identical=True,
                persistent_sessions_precede_trace_capture=True,
                checkpoints=[
                    dict(step=step, all_values_finite=True, recurrent_conv_projected_sha256=[["a" * 64] * 4] * 3)
                    for step in changing_input_checkpoints(4096)
                ],
            )
            for batch in (16, 32)
        ],
    )


def test_quick_gate_stays_64_and_long_gate_includes_4096():
    assert changing_input_checkpoints(64) == (1, 2, 4, 8, 16, 32, 64)
    assert changing_input_checkpoints(4096)[-1] == 4096
    result = validate_long_horizon(report())
    assert result["exact_boundary_comparison_passed"]
    assert not result["independent_dense_reference"] and not result["full_model_qualified"]


@pytest.mark.parametrize("updates", [True, 0, 65, 4096.0, 8192])
def test_unknown_horizon_rejected(updates, expect_error):
    with expect_error(ValueError, "64 or 4096"):
        changing_input_checkpoints(updates)


@pytest.mark.parametrize(
    "corruption", ["short", "missing_rank", "missing_state", "nonfinite", "aliased", "missing_batch"]
)
def test_incomplete_or_invalid_boundary_evidence_rejected(corruption, expect_error):
    value = report()
    case = value["cases"][0]
    if corruption == "short":
        case["checkpoints"].pop()
    elif corruption == "missing_rank":
        case["checkpoints"][-1]["recurrent_conv_projected_sha256"][0] = ["a" * 64] * 3
    elif corruption == "missing_state":
        case["checkpoints"][-1]["recurrent_conv_projected_sha256"].pop()
    elif corruption == "nonfinite":
        case["checkpoints"][-1]["all_values_finite"] = False
    elif corruption == "aliased":
        case["persistent_sessions_precede_trace_capture"] = False
    else:
        value["cases"].pop()
    with expect_error(ValueError, "Long-horizon"):
        validate_long_horizon(value)


def test_combined_boundary_requires_its_explicit_policy_pair(expect_error):
    value = report()
    value.update(baseline=CANDIDATE, candidate=COMBINED_GDN_POLICY)
    with expect_error(ValueError, "policies differ"):
        validate_long_horizon(value)
    assert validate_long_horizon(value, baseline=CANDIDATE, candidate=COMBINED_GDN_POLICY)[
        "exact_boundary_comparison_passed"
    ]
    with expect_error(ValueError, "Unsupported"):
        validate_long_horizon(value, baseline=BASELINE, candidate=COMBINED_GDN_POLICY)
