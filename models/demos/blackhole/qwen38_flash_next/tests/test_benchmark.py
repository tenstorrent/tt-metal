# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import pytest

from models.demos.blackhole.qwen38_flash_next.tools.benchmark import answer, summarize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("work\n#### 1,234", Decimal(1234)),
        ("#### -0.50.", Decimal("-.5")),
        ("#### 1\ncorrected\n#### 2", Decimal(2)),
        ("42", None),
        ("#### 1/2", None),
        ("#### 42 or 43", None),
        ("#### 1e3", None),
        ("####", None),
    ],
)
def test_answer_is_explicit_and_unambiguous(text, expected):
    assert answer(text) == expected


def test_incomplete_run_cannot_pass():
    rows = [
        {
            "correct": True,
            "ttft_s": 1,
            "stream_decode_tokens_per_s": 10,
            "finish_reason": "stop",
            "usage": {"completion_tokens": 11},
        }
    ]
    summary = summarize(rows, 64, "2026-09-29T00:00:00+00:00", 2)
    assert summary["accuracy"] == 1
    assert summary["completed_samples"] == 1
    assert summary["dataset_samples"] == 1319
    assert summary["aggregate_output_tokens_per_s"] == 5.5
    assert not summary["passed"]
