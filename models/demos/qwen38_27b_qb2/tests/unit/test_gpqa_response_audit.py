# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Saved-output diagnosis must preserve scores, provenance and dataset privacy."""

import hashlib
import json

import pytest

from models.demos.qwen38_27b_qb2.tests.gpqa_response_audit import audit, qualify


def fixtures(tmp_path):
    raw_dir = tmp_path / "private"
    raw_dir.mkdir()
    rows = []
    for i, (finish, prompt, output, text, reasoning, correct) in enumerate(
        (
            ("length", 10, 64, "", "still reasoning without a final answer", 0),
            ("stop", 10, 9, "PRIVATE_CORRECT_ANSWER", "private correct reasoning", 1),
            ("length", 210, 46, "", "private context-limited reasoning", 0),
        )
    ):
        raw = dict(
            finish_reason=finish,
            usage=dict(prompt_tokens=prompt, completion_tokens=output),
            text=text,
            reasoning=reasoning,
        )
        (raw_dir / f"{i:03d}.json").write_text(json.dumps(raw))
        rows.append(
            dict(
                id=i,
                finish_reason=finish,
                usage=raw["usage"],
                correct=correct,
                final_answer_sha256=hashlib.sha256(text.encode()).hexdigest(),
                reasoning_sha256=hashlib.sha256(reasoning.encode()).hexdigest(),
            )
        )
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text("\n".join(map(json.dumps, rows)) + "\n")
    return receipts, raw_dir, rows


def test_all_questions_stay_scored_and_distinct_limits_are_identified(tmp_path):
    receipts, raw_dir, _ = fixtures(tmp_path)
    report = audit(receipts, raw_dir, count=3, max_output_tokens=64, max_model_len=256)
    assert report["full_score"] == {"correct": 1, "total": 3}
    assert report["completed_response_score"] == {"correct": 1, "total": 3}
    assert report["incomplete_credited_by_harness"] == 0
    assert report["limits"] == {"output_budget": 1, "not_length_limited": 1, "model_context": 1}
    assert report["groups"]["natural_stop"]["count"] == 1
    assert report["groups"]["length_limited"]["empty_final"] == 2
    assert "PRIVATE_CORRECT_ANSWER" not in json.dumps(report)
    assert "private correct reasoning" not in json.dumps(report)


@pytest.mark.parametrize("finish", ["length", "other"])
def test_incomplete_parser_matches_never_enter_completion_qualified_score(tmp_path, finish):
    receipts, raw_dir, rows = fixtures(tmp_path)
    path = raw_dir / "000.json"
    raw = json.loads(path.read_text())
    raw.update(finish_reason=finish, text="PRIVATE_PARSER_MATCH_BEFORE_CUTOFF")
    path.write_text(json.dumps(raw))
    rows[0].update(
        correct=1, finish_reason=finish, final_answer_sha256=hashlib.sha256(raw["text"].encode()).hexdigest()
    )
    receipts.write_text("\n".join(map(json.dumps, rows)))
    report = audit(receipts, raw_dir, count=3, max_output_tokens=64, max_model_len=256)
    assert report["full_score"] == {"correct": 2, "total": 3}
    assert report["completed_response_score"] == {"correct": 1, "total": 3}
    assert report["incomplete_credited_by_harness"] == 1
    assert "PRIVATE_PARSER_MATCH" not in json.dumps(report)


@pytest.mark.parametrize("field,value", [("text", "replacement"), ("finish_reason", "stop")])
def test_different_unscored_output_is_rejected(tmp_path, field, value, expect_error):
    receipts, raw_dir, _ = fixtures(tmp_path)
    path = raw_dir / "000.json"
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw))
    with expect_error(ValueError, "scored receipt"):
        audit(receipts, raw_dir, count=3, max_output_tokens=64, max_model_len=256)


@pytest.mark.parametrize("failure", ["missing_raw", "duplicate_receipt"])
def test_incomplete_or_duplicate_selection_is_rejected(tmp_path, failure, expect_error):
    receipts, raw_dir, rows = fixtures(tmp_path)
    if failure == "missing_raw":
        (raw_dir / "002.json").unlink()
    else:
        receipts.write_text("\n".join(map(json.dumps, [rows[0], rows[1], rows[1]])))
    with expect_error(ValueError, "complete selected set|exactly once"):
        audit(receipts, raw_dir, count=3, max_output_tokens=64, max_model_len=256)


def test_wrong_launch_limits_are_rejected(tmp_path, expect_error):
    receipts, raw_dir, _ = fixtures(tmp_path)
    with expect_error(ValueError, "declared configuration"):
        audit(receipts, raw_dir, count=3, max_output_tokens=32, max_model_len=256)


def qualification_fixture():
    result = dict(
        completed_samples=198,
        dataset_samples=198,
        full_dataset=True,
        max_output_tokens=65536,
        accuracy_threshold=0.892,
        correct=177,
        accuracy=177 / 198,
        passed=True,
        truncated_samples=1,
        truncated_correct=1,
    )
    report = dict(
        selected_count=198,
        max_output_tokens=65536,
        max_model_len=262144,
        full_score=dict(correct=177, total=198),
        completed_response_score=dict(correct=176, total=198),
        incomplete_credited_by_harness=1,
        groups=dict(length_limited=dict(count=1, correct=1)),
    )
    return {"gpqa_result": result}, report


def test_truncated_answer_cannot_supply_the_last_qualification_point():
    summary, report = qualification_fixture()
    result = qualify(summary, report)
    assert result["harness_passed"] and result["harness_correct"] == 177
    assert not result["completed_response_passed"] and result["completed_response_correct"] == 176
    assert result["selected_count"] == 198


def test_complete_177_answer_gate_remains_unchanged():
    summary, report = qualification_fixture()
    summary["gpqa_result"]["truncated_correct"] = 0
    report["groups"]["length_limited"]["correct"] = 0
    report["completed_response_score"]["correct"] = 177
    report["incomplete_credited_by_harness"] = 0
    assert qualify(summary, report)["completed_response_passed"]


@pytest.mark.parametrize(
    "field,value",
    [("correct", 178), ("accuracy_threshold", 0.85), ("completed_samples", 197), ("truncated_correct", 0)],
)
def test_changed_or_inconsistent_published_summary_cannot_qualify(field, value, expect_error):
    summary, report = qualification_fixture()
    summary["gpqa_result"][field] = value
    with expect_error(ValueError, "unchanged|disagrees"):
        qualify(summary, report)
