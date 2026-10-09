# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from models.demos.qwen38_27b_qb2.tests.tau_verified_benchmark import summarize, validate_endpoint


@pytest.mark.parametrize(
    "endpoint", ["https://user:secret@example.com/v1", "https://example.com/v1?key=secret", "file:///tmp/api"]
)
def test_credentials_never_belong_in_recorded_endpoint(expect_error, endpoint):
    with expect_error(ValueError, "Endpoint must"):
        validate_endpoint(endpoint)


def test_incomplete_run_cannot_be_reported_as_reference_accuracy(tmp_path):
    (tmp_path / "official.json").write_text(
        json.dumps({"simulations": [{"task_id": "0", "reward_info": {"reward": 1}}]})
    )
    (tmp_path / "calls").mkdir()
    (tmp_path / "calls/a.json").write_text('{"response":{"choices":[]}}')
    report = summarize(tmp_path, ["0", "1"], 1, True)
    assert report["accuracy"] is None and not report["complete"]
    assert report["lower_bound_counting_missing_as_incorrect"] == 0.5
    assert report["selected"] == 2 and report["completed"] == 1


def test_duplicate_results_rejected(tmp_path, expect_error):
    (tmp_path / "official.json").write_text(json.dumps({"simulations": [{"task_id": "0"}, {"task_id": "0"}]}))
    with expect_error(ValueError, "Duplicate or unexpected task results"):
        summarize(tmp_path, ["0", "1"], 0, False)


def test_completed_failures_remain_in_denominator(tmp_path):
    (tmp_path / "calls").mkdir()
    (tmp_path / "calls/a.json").write_text('{"response":{"choices":[]}}')
    (tmp_path / "official.json").write_text(
        json.dumps(
            {
                "simulations": [
                    {"task_id": "0", "reward_info": {"reward": 1}},
                    {"task_id": "1", "reward_info": {"reward": 0}},
                ]
            }
        )
    )
    report = summarize(tmp_path, ["0", "1"], 0, False)
    assert report["complete"] and report["accuracy"] == 0.5
    assert not report["agentic_release_qualified"]
