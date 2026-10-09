# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A response audit must wait for the exact completed measured invocation."""

import pytest

from models.demos.qwen38_27b_qb2.demo.run_gpqa_audit_followup import predecessor_ready


def completed():
    service = dict(LoadState="loaded", ActiveState="inactive", MainPID="0", Result="success", InvocationID="expected")
    receipt = dict(
        state="completed",
        passed=False,
        steps=[dict(name=name, state="completed") for name in ("native-g0-run", "native-control")],
        **{"native-control": dict(owned_processes_stopped=True)},
    )
    return service, receipt


def test_failed_score_is_a_completed_experiment():
    service, receipt = completed()
    assert predecessor_ready(service, receipt, "expected")


def test_collected_unit_requires_complete_receipt():
    service, receipt = completed()
    service.update(LoadState="not-found", InvocationID="")
    assert predecessor_ready(service, receipt, "expected")


def test_stale_file_cannot_override_live_process():
    service, receipt = completed()
    service.update(ActiveState="active", MainPID="123")
    assert not predecessor_ready(service, receipt, "expected")


@pytest.mark.parametrize(
    "fault", ["different_invocation", "failed_service", "missing_receipt", "missing_stage", "workers_live"]
)
def test_failed_or_replaced_predecessor_cannot_authorize_audit(fault, expect_error):
    service, receipt = completed()
    if fault == "different_invocation":
        service["InvocationID"] = "replaced"
    elif fault == "failed_service":
        service["Result"] = "timeout"
    elif fault == "missing_receipt":
        receipt = None
    elif fault == "missing_stage":
        receipt["steps"].pop()
    elif fault == "workers_live":
        receipt["native-control"]["owned_processes_stopped"] = False
    with expect_error(ValueError, "replaced|exit|complete"):
        predecessor_ready(service, receipt, "expected")
