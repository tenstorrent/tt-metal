# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prevent teacher-forcing or final-norm misalignment from impersonating model error."""

from types import SimpleNamespace

import pytest

from models.demos.qwen38_27b_qb2.tests.reference_comparison import validate_teacher_forcing, vector_metrics


class TensorFixture:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def tolist(self):
        return self.values

    def argmax(self, dim):
        assert dim == -1 and self.shape[0] == 1
        return SimpleNamespace(item=lambda: max(range(self.shape[1]), key=lambda i: self.values[0][i]))


def reference():
    return [
        dict(
            step=i,
            input_ids=TensorFixture([ids]),
            layer_last_hidden=[TensorFixture([[1, 2, 3]]) for _ in range(3)],
            logits=TensorFixture([[0, 1, 2, 3, 4]]),
        )
        for i, ids in enumerate(([1, 2], [4], [4]))
    ]


def validate(rows):
    return validate_teacher_forcing(rows, [1, 2], layers=2, hidden_size=3, vocab_size=5)


def test_teacher_forcing_accounts_for_prompt_and_decode_positions():
    assert validate(reference()) == 4


@pytest.mark.parametrize("failure", ["token", "step", "missing_norm", "wrong_width", "wrong_vocab"])
def test_bad_alignment_is_rejected_before_hardware(failure, expect_error):
    rows = reference()
    if failure == "token":
        rows[1]["input_ids"] = TensorFixture([[3]])
    elif failure == "step":
        rows[1]["step"] = 2
    elif failure == "missing_norm":
        rows[0]["layer_last_hidden"].pop()
    elif failure == "wrong_width":
        rows[0]["layer_last_hidden"][0] = TensorFixture([[1, 2]])
    else:
        rows[0]["logits"] = TensorFixture([[1, 2]])
    with expect_error(ValueError, "Reference"):
        validate(rows)


def test_metrics_distinguish_scale_error_from_correlation():
    metrics = vector_metrics([2, 4, 6], [1, 2, 3])
    assert metrics["pcc"] == pytest.approx(1)
    assert metrics["cosine"] == pytest.approx(1)
    assert metrics["relative_rms_error"] == pytest.approx(1)
    assert metrics["max_absolute_error"] == 3


def test_constant_and_zero_references_do_not_claim_correlation():
    metrics = vector_metrics([0, 0], [0, 0])
    assert metrics["rms_error"] == 0
    assert metrics["relative_rms_error"] is None
    assert metrics["pcc"] is None
    assert metrics["cosine"] is None
    assert vector_metrics([1, 1], [1, 1])["pcc"] is None


@pytest.mark.parametrize("actual,expected", [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])])
def test_invalid_comparisons_are_rejected(actual, expected, expect_error):
    with expect_error(ValueError, "Comparison"):
        vector_metrics(actual, expected)
