# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify scheduling covers every user/head exactly once across uneven waves."""

import pytest

from models.demos.qwen38_27b_qb2.experiments.gdn_step.op import work_items


@pytest.mark.parametrize("heads", [1, 12, 96, 120, 121, 192, 768])
@pytest.mark.parametrize("splits", [1, 2, 4])
def test_disjoint_complete_work(heads, splits):
    assignments = work_items(heads, 12, 10, splits)
    covered = [first + i * stride for _, _, first, stride, count in assignments for i in range(count)]
    assert sorted(covered) == list(range(heads * splits))
    assert len({(x, y) for x, y, *_ in assignments}) == len(assignments)
    assert all(0 <= x < 12 and 0 <= y < 10 and count > 0 for x, y, _, _, count in assignments)
    assert max(a[-1] for a in assignments) - min(a[-1] for a in assignments) <= 1
    # Map the scheduled partitions to their physical state tiles and output
    # segments. Every byte range belongs to exactly one worker, including the
    # last partial wave, and each 128-wide K reduction stays inside one worker.
    state_tiles, output_segments = [], []
    columns = 4 // splits
    for work in covered:
        head, partition = divmod(work, splits)
        for vc in range(partition * columns, (partition + 1) * columns):
            state_tiles.extend(head * 16 + kr * 4 + vc for kr in range(4))
            output_segments.append(head * 4 + vc)
    assert sorted(state_tiles) == list(range(heads * 16))
    assert sorted(output_segments) == list(range(heads * 4))


@pytest.mark.parametrize("args", [(0, 12, 10), (-1, 12, 10), (12, 0, 10), (12, 12, 0)])
def test_invalid_geometry(args, expect_error):
    with expect_error(ValueError, "must be positive"):
        work_items(*args)


@pytest.mark.parametrize("splits", [0, -1, 3, 8, 2.0, True])
def test_reject_unsupported_value_partition(splits, expect_error):
    with expect_error(ValueError, "Value splits"):
        work_items(12, 12, 10, splits)


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
