# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unbounded launch specs and preserve the requested measurement axes."""

import json

import pytest

from models.demos.qwen38_27b_qb2.demo.overnight_plan import (
    load_followup,
    optional_capacity_result,
    perf_cases,
    remaining_stage_seconds,
)
from models.demos.qwen38_27b_qb2.demo.release_queue_after_capacity import validate
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan


@pytest.mark.parametrize(
    "command,timeout",
    [("shell command", 60), (["python"], 60), (["/python"], 0), (["/python"], True), (["/python"], 7201)],
)
def test_invalid_evaluator_spec(tmp_path, expect_error, command, timeout):
    path = tmp_path / "followup.json"
    path.write_text(json.dumps(dict(command=command, timeout_seconds=timeout)))
    with expect_error(ValueError, "Follow-up"):
        load_followup(path)


def test_new_stage_keeps_shutdown_reserve():
    assert remaining_stage_seconds(1000, 0, 900) == 760
    assert remaining_stage_seconds(1000, 200, 900) == 0


def test_planned_cells_fit_existing_capacity_guard():
    cases = perf_cases()
    assert cases[0]["cells"][0] == (32768, 8)
    assert {row["token_budget"] for row in cases} == {16384, 32768, 65536}
    for case in cases:
        for length, batch in case["cells"]:
            assert make_plan(1, batches=(batch,), input_lengths=(length,))["cells"][0]["status"] == "queued"


def capacity_receipt():
    return dict(
        state="allocation_failed",
        cleanup_completed=True,
        cells=[dict(status="oom")],
        error=dict(
            type="RuntimeError",
            message="Out of Memory: Not enough space to allocate 570425344 B DRAM buffer across 8 banks",
        ),
    )


def test_capacity_limit_is_narrow_and_keeps_hardware_failures_fatal():
    receipt = capacity_receipt()
    assert optional_capacity_result("bfp8-budget64k", receipt, 1)
    assert not optional_capacity_result("bfp8-budget32k", receipt, 1)
    assert not optional_capacity_result("bfp8-budget64k", receipt, 137)
    assert not optional_capacity_result("bfp8-budget64k", dict(receipt, cleanup_completed=False), 1)
    assert not optional_capacity_result(
        "bfp8-budget64k", dict(receipt, error=dict(type="RuntimeError", message="Dispatch timeout")), 1
    )


def test_release_requires_stopped_exact_unit_and_clean_optional_oom(expect_error):
    props = dict(InvocationID="exact", MainPID="0", ActiveState="failed", Result="exit-code", ExecMainStatus="1")
    queue = dict(
        state="failed",
        active_stage="bfp8-budget64k",
        stages=[dict(name="qualification", state="completed"), dict(name="bfp8-budget64k", state="failed")],
    )
    validate(props, queue, capacity_receipt(), "exact")
    for changed in (dict(props, MainPID="123"), dict(props, InvocationID="other"), dict(props, ExecMainStatus="137")):
        with expect_error(ValueError, "exact failed, stopped"):
            validate(changed, queue, capacity_receipt(), "exact")
    with expect_error(ValueError, "completed preceding stages"):
        validate(
            props,
            dict(queue, stages=[dict(name="qualification", state="failed"), queue["stages"][-1]]),
            capacity_receipt(),
            "exact",
        )
