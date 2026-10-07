# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify scheduling covers every user/head exactly once across uneven waves."""

import pytest

from models.demos.qwen38_27b_qb2.experiments.gdn_step.op import work_items


@pytest.mark.parametrize("heads", [1, 12, 96, 120, 121, 192, 768])
def test_disjoint_complete_work(heads):
    assignments = work_items(heads, 12, 10)
    covered = [first + i * stride for _, _, first, stride, count in assignments for i in range(count)]
    assert sorted(covered) == list(range(heads))
    assert len({(x, y) for x, y, *_ in assignments}) == len(assignments)
    assert all(0 <= x < 12 and 0 <= y < 10 and count > 0 for x, y, _, _, count in assignments)
    assert max(a[-1] for a in assignments) - min(a[-1] for a in assignments) <= 1


@pytest.mark.parametrize("args", [(0, 12, 10), (-1, 12, 10), (12, 0, 10), (12, 12, 0)])
def test_invalid_geometry(args, expect_error):
    with expect_error(ValueError, "must be positive"):
        work_items(*args)


def test_accuracy_checks_each_head_and_rejects_scaling():
    import torch

    from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy

    expected = torch.randn(768, 128, generator=torch.Generator().manual_seed(42))
    assert accuracy(expected, expected)["passed"]
    damaged = expected.clone()
    damaged[-1] *= -1
    assert not accuracy(damaged, expected)["passed"]
    assert not accuracy(expected * 1.1, expected)["passed"]
    damaged[-1, 0] = float("nan")
    assert not accuracy(damaged, expected)["passed"]
