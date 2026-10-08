# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_shared_qk import compare


def fixture():
    return [
        dict(
            batch=32,
            shared_qk=shared,
            passed=True,
            input_sha256=["q", "k", "v", "g", "b", "state"],
            state_sha256_per_rank=["s"] * 4,
            output_sha256_per_rank=["o"] * 4,
            traced_call_us=[time] * 5,
        )
        for shared, time in ((False, 100), (True, 80), (False, 100))
    ]


def test_shared_speedup_requires_matched_numerics_and_stable_control():
    result = compare(fixture())
    assert result["qualified_speedup"] == 1.25
    assert result["output_and_state_bit_identical"]
    assert not result["promoted_to_model"]


@pytest.mark.parametrize("failure", ["order", "batch", "input", "state", "output", "rank", "accuracy", "sample", "nan"])
def test_shared_comparison_rejects_incomplete_or_unmatched_evidence(failure, expect_error):
    cases = fixture()
    row = cases[1]
    if failure == "order":
        row["shared_qk"] = False
    elif failure == "batch":
        row["batch"] = 16
    elif failure == "input":
        row["input_sha256"][0] = "different"
    elif failure == "state":
        row["state_sha256_per_rank"][0] = "different"
    elif failure == "output":
        row["output_sha256_per_rank"][0] = "different"
    elif failure == "rank":
        row["output_sha256_per_rank"].pop()
    elif failure == "accuracy":
        row["passed"] = False
    elif failure == "sample":
        row["traced_call_us"].pop()
    elif failure == "nan":
        row["traced_call_us"][0] = float("nan")
    with expect_error(ValueError, "controls|Mismatched|numerical|timing"):
        compare(cases)


def test_timing_drift_cannot_qualify_speedup():
    cases = fixture()
    cases[2]["traced_call_us"] = [104] * 5
    result = compare(cases)
    assert not result["timing_comparison_qualified"]
    assert result["qualified_speedup"] is None


def test_correct_but_slower_candidate_is_preserved():
    cases = fixture()
    cases[1]["traced_call_us"] = [125] * 5
    assert compare(cases)["qualified_speedup"] == 0.8
