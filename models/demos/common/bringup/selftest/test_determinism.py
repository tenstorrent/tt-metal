# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""testing/determinism.py: A, B, A and repeats, compared bit for bit."""

import pytest
import torch

from models.demos.common.bringup.testing import determinism as D


def _stream(outs):
    it = iter(outs)
    return lambda: next(it)


def test_identical_runs_pass_and_nested_outputs_are_compared():
    a = torch.randn(4, 8, dtype=torch.bfloat16)
    assert D.run(lambda: (a.clone(), {"ids": torch.arange(3)}), lambda: None, n=5) is None


def test_a_change_right_after_b_is_named():
    a = torch.randn(4, 8)
    bad = a.clone()
    bad[1, 2] += 1e-6
    err = D.run(_stream([a, bad, a, a, a]), lambda: None, n=5)
    assert "run 2 of 5 (right after B)" in err and "1/32" in err


def test_a_late_change_is_caught():
    a = torch.randn(16)
    bad = a.clone()
    bad[-1] = float("nan")
    assert "run 5 of 5" in D.run(_stream([a, a, a, a, bad]), lambda: None, n=5)


def test_signed_zero_counts_as_different():
    assert D.run(_stream([torch.zeros(3), -torch.zeros(3)]), lambda: None, n=2)


def test_b_runs_once_between_the_first_and_second_a():
    order = []
    a = torch.ones(2)

    def run_a():
        order.append("A")
        return a

    D.run(run_a, lambda: order.append("B"), n=3)
    assert order == ["A", "B", "A", "A"]


def test_first_given_is_not_rerun_and_n_off_skips():
    calls = []
    D.run(lambda: calls.append(1) or torch.ones(1), lambda: None, first=torch.ones(1), n=3)
    assert len(calls) == 2
    assert D.run(lambda: 1 / 0, lambda: None, n=0) is None


def test_assert_fails_with_the_difference():
    with pytest.raises(AssertionError, match="not deterministic"):
        D.assert_deterministic(_stream([torch.zeros(2), torch.ones(2)]), lambda: None, n=2, label="x")


def test_default_repeats_come_from_defaults_yaml():
    assert D.repeats() == 5
