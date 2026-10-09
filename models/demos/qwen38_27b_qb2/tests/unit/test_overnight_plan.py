# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unbounded launch specs and preserve the requested measurement axes."""

import json

import pytest

from models.demos.qwen38_27b_qb2.demo.overnight_plan import load_followup, perf_cases, remaining_stage_seconds
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
