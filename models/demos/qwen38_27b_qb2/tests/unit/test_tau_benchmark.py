# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep failed/missing trials in the denominator and retain malformed calls."""

import json

import pytest

from models.demos.qwen38_27b_qb2.tests.tau_benchmark import TASK_IDS, call_statistics, summarize


@pytest.mark.parametrize(
    "returncode,timed_out,termination,passed",
    [
        (0, False, "user_stop", True),
        (1, False, "user_stop", False),
        (0, True, "user_stop", False),
        (0, False, "timeout", False),
    ],
)
def test_reward_cannot_hide_timeout_or_failed_process(tmp_path, returncode, timed_out, termination, passed):
    task = TASK_IDS[0]
    directory = tmp_path / task
    directory.mkdir()
    (directory / "raw-calls.jsonl").write_text('{"response":{"choices":[]}}\n')
    (directory / "completed-result.json").write_text(
        json.dumps(
            {"simulations": [{"task_id": task, "reward_info": {"reward": 1}, "termination_reason": termination}]}
        )
    )
    outcomes = {task: dict(returncode=returncode, timed_out=timed_out)}
    progress = summarize(tmp_path, outcomes)
    assert progress["accuracy"] is None
    report = summarize(tmp_path, outcomes, final=True)
    assert report["passed"] == int(passed)
    assert report["accuracy"] == int(passed) / 12
    assert report["selected"] == 12 and report["attempted"] == 1
    assert not report["all_tasks_attempted"]


def test_raw_tool_arguments_are_classified_without_repair(tmp_path):
    path = tmp_path / "raw.jsonl"
    tool = lambda arguments: {"function": {"name": "lookup", "arguments": arguments}}
    row = {
        "response": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"tool_calls": [tool('{"a": 1}'), tool('{"broken"'), tool("[]")]},
                }
            ]
        }
    }
    path.write_text(json.dumps(row) + '\n{"error_type":"TimeoutError"}\n{"partial":')
    stats = call_statistics(path)
    assert stats == dict(calls=2, tool_calls=3, malformed_tool_calls=2, truncated_calls=1, call_errors=2)
    assert '"partial"' in path.read_text()


def test_wrong_task_artifact_never_counts_as_success(tmp_path):
    directory = tmp_path / TASK_IDS[0]
    directory.mkdir()
    (directory / "completed-result.json").write_text(
        json.dumps({"simulations": [{"task_id": "different", "reward_info": {"reward": 1}}]})
    )
    report = summarize(tmp_path, {TASK_IDS[0]: dict(returncode=0, timed_out=False)}, final=True)
    assert report["passed"] == 0 and "artifact_error" in report["trials"][0]


def test_no_inference_is_a_setup_failure_not_zero_model_accuracy(tmp_path):
    report = summarize(tmp_path, {task: dict(returncode=1, timed_out=False) for task in TASK_IDS}, final=True)
    assert report["state"] == "setup_failed"
    assert report["accuracy"] is None and report["attempted"] == 12
