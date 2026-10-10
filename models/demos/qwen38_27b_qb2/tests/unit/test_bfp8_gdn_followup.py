# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from models.demos.qwen38_27b_qb2.demo.run_bfp8_gdn_followup import (
    FUSION_POLICIES,
    POLICIES,
    control_stable,
    predecessor_ready,
    validate_policies,
)


def terminal():
    return dict(MainPID="0", ActiveState="inactive", LoadState="loaded", Result="success", InvocationID="expected")


def receipt():
    return dict(
        state="completed",
        passed=True,
        api_and_tool_smoke_passed=True,
        owned_container_removed=True,
        evaluation_exit_code=0,
    )


def test_live_predecessor_blocks_even_with_a_completed_receipt():
    props = dict(terminal(), MainPID="123", ActiveState="active")
    assert not predecessor_ready(props, receipt(), "expected")


@pytest.mark.parametrize(
    "field", ["state", "passed", "api_and_tool_smoke_passed", "owned_container_removed", "evaluation_exit_code"]
)
def test_incomplete_container_release_cannot_start_hardware(field, expect_error):
    data = receipt()
    data.pop(field)
    with expect_error(ValueError, "lacks successful qualification"):
        predecessor_ready(terminal(), data, "expected")


def test_changed_or_failed_predecessor_rejected(expect_error):
    with expect_error(ValueError, "invocation changed"):
        predecessor_ready(dict(terminal(), InvocationID="different"), receipt(), "expected")
    with expect_error(ValueError, "exited unsuccessfully"):
        predecessor_ready(dict(terminal(), Result="timeout"), receipt(), "expected")


def test_clean_terminal_or_collected_unit_accepted():
    assert predecessor_ready(terminal(), receipt(), "expected")
    assert predecessor_ready(dict(terminal(), LoadState="not-found", InvocationID=""), receipt(), "expected")


@pytest.mark.parametrize("fusion", [False, True])
def test_candidate_retains_qualified_precision(tmp_path, expect_error, fusion):
    model = Path(__file__).resolve().parents[2]
    validate_policies(model, fusion=fusion)
    (tmp_path / "config").mkdir()
    policies = FUSION_POLICIES if fusion else POLICIES
    for name in policies.values():
        (tmp_path / "config" / name).write_bytes((model / "config" / name).read_bytes())
    path = tmp_path / "config" / policies["shared-qk"]
    data = json.loads(path.read_text())
    data["weight_groups"]["attention"] = "bfloat4_b"
    path.write_text(json.dumps(data))
    with expect_error(ValueError, "changes precision"):
        validate_policies(tmp_path, fusion=fusion)


def test_layer_gate_requires_every_rank_and_batch_even_with_passing_summary(expect_error):
    from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import BATCHES, compare
    from models.demos.qwen38_27b_qb2.tests.unit.test_gdn_epilogue_layer import cases

    rows = [dict(row, batch=batch) for batch in BATCHES for row in cases()]
    report = dict(
        state="completed",
        passed=True,
        cleanup_completed=True,
        candidate_recurrence="single_step_flat_prepare_epilogue",
        cases=rows,
        comparisons=[compare(rows[i : i + 3]) for i in range(0, len(rows), 3)],
    )
    assert predecessor_ready(terminal(), report, "expected", layer_release=True)
    assert not predecessor_ready(dict(terminal(), MainPID="123"), report, "expected", layer_release=True)
    report["cases"][1] = dict(report["cases"][1], state_sha256_per_rank=["changed"] * 4)
    with expect_error(ValueError, "changed"):
        predecessor_ready(terminal(), report, "expected", layer_release=True)
    report["cases"].pop()
    with expect_error(ValueError, "Incomplete"):
        predecessor_ready(terminal(), report, "expected", layer_release=True)


def test_control_drift_or_changed_tokens_does_not_qualify_speedup():
    assert control_stable({"cells": [dict(same_output_hash_as_native=True, decode_uplift_percent=2.9)]})
    assert not control_stable({"cells": [dict(same_output_hash_as_native=True, decode_uplift_percent=-3.1)]})
    assert not control_stable({"cells": [dict(same_output_hash_as_native=False, decode_uplift_percent=0)]})
    assert not control_stable({"cells": []})


def release_receipt():
    return dict(
        state="completed",
        cleanup_completed=True,
        owned_container_removed=True,
        device_reset_required=True,
        hardware_health_proven=False,
        reason="performance_priority",
    )


def test_explicit_priority_release_is_not_an_eval_pass(expect_error):
    data = release_receipt()
    assert predecessor_ready(terminal(), data, "expected", clean_release=True)
    with expect_error(ValueError, "lacks successful qualification"):
        predecessor_ready(terminal(), data, "expected")
    assert not predecessor_ready(
        dict(terminal(), MainPID="1", ActiveState="active"), data, "expected", clean_release=True
    )


@pytest.mark.parametrize(
    "field",
    [
        "state",
        "cleanup_completed",
        "owned_container_removed",
        "device_reset_required",
        "hardware_health_proven",
        "reason",
    ],
)
def test_priority_release_requires_complete_audit(field, expect_error):
    data = release_receipt()
    data.pop(field)
    with expect_error(ValueError, "lacks audited performance-priority release"):
        predecessor_ready(terminal(), data, "expected", clean_release=True)
