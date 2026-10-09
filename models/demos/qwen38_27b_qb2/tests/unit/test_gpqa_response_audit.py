# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Saved-output diagnosis must preserve scores, provenance and dataset privacy."""

import hashlib
import json

import pytest

from models.demos.qwen38_27b_qb2.tests.gpqa_response_audit import audit


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
    assert report["limits"] == {"output_budget": 1, "not_length_limited": 1, "model_context": 1}
    assert report["groups"]["natural_stop"]["count"] == 1
    assert report["groups"]["length_limited"]["empty_final"] == 2
    assert "PRIVATE_CORRECT_ANSWER" not in json.dumps(report)
    assert "private correct reasoning" not in json.dumps(report)


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
