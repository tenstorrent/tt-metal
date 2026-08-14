# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Correctness probe for the ragged reduce paths this branch changes.

Why this file exists: the branch ADDS
test_softmax_large_algorithm_not_multiple_of_32_for_dim_hw to the shipped suite --
i.e. ragged reduce dimensions on the _large softmax kernels were previously
UNTESTED. Before treating a perf delta on those shapes as a win, we need to know
whether the merge-base was actually correct there. Otherwise the comparison is
"right vs wrong", not "fast vs slow".

This file is untracked, so it survives `git checkout` of either commit and can be
run unchanged against both:

    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/moreh_mean_perf/check_ragged_correctness.py

Real value comparisons against torch -- not just finiteness.
"""

import pytest
import torch

import ttnn

TILE = 32
STRAT = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy
BSTRAT = ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy

RTOL = ATOL = 0.05


def ragged(nt):
    return (nt - 1) * TILE + 17


def to_dev(t, device):
    return ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=float("nan"))


# (id, shape, dim, strategy)
SOFTMAX_CASES = [
    ("small_h_ragged", [1, 7, ragged(4), 256], 2, STRAT.SMALL_H),
    ("small_w_ragged", [1, 7, 256, ragged(4)], 3, STRAT.SMALL_W),
    ("large_h_ragged", [1, 7, ragged(32), 256], 2, STRAT.LARGE_H),
    ("large_w_ragged", [1, 7, 256, ragged(32)], 3, STRAT.LARGE_W),
    # single-tile ragged: the case that used to take a separate Wt==1 / Ht==1 branch
    ("large_h_ragged_1tile", [1, 7, ragged(1), 256], 2, STRAT.LARGE_H),
    ("large_w_ragged_1tile", [1, 7, 256, ragged(1)], 3, STRAT.LARGE_W),
]


@pytest.mark.parametrize("case", SOFTMAX_CASES, ids=[c[0] for c in SOFTMAX_CASES])
def test_moreh_softmax_ragged(case, device):
    cid, shape, dim, strategy = case
    torch.manual_seed(2024)
    x = torch.rand(shape, dtype=torch.float32)
    expected = torch.softmax(x, dim=dim)
    got = ttnn.to_torch(ttnn.operations.moreh.softmax(to_dev(x, device), dim, strategy=strategy))
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "case",
    [
        ("bw_small_h_ragged", [1, 7, ragged(4), 256], 2, BSTRAT.SMALL_H),
        ("bw_small_w_ragged", [1, 7, 256, ragged(4)], 3, BSTRAT.SMALL_W),
    ],
    ids=["bw_small_h_ragged", "bw_small_w_ragged"],
)
def test_moreh_softmax_backward_ragged(case, device):
    cid, shape, dim, strategy = case
    torch.manual_seed(2024)
    x = torch.rand(shape, dtype=torch.float32, requires_grad=True)
    y = torch.softmax(x, dim=dim)
    dy = torch.rand(shape, dtype=torch.float32)
    y.backward(dy)
    expected = x.grad

    got = ttnn.to_torch(
        ttnn.operations.moreh.softmax_backward(to_dev(y.detach(), device), to_dev(dy, device), dim, strategy=strategy)
    )
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("nt", [1, 4, 32], ids=["ht1", "ht4", "ht32"])
def test_moreh_sum_h_ragged(nt, device):
    torch.manual_seed(2024)
    h = ragged(nt)
    x = torch.rand([1, 7, h, 256], dtype=torch.float32)
    expected = torch.sum(x, dim=2, keepdim=True)
    got = ttnn.to_torch(ttnn.operations.moreh.sum(to_dev(x, device), 2, keepdim=True))
    # sums over up to 1009 bf16 rows: loosen to a relative check
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=0.05, atol=0.5)


@pytest.mark.parametrize("nt", [4, 32], ids=["wt4", "wt32"])
def test_layernorm_ragged(nt, device):
    torch.manual_seed(2024)
    w = ragged(nt)
    x = torch.rand([1, 1, 256, w], dtype=torch.float32)
    expected = torch.nn.functional.layer_norm(x, (w,), eps=1e-5)
    got = ttnn.to_torch(ttnn.layer_norm(to_dev(x, device), epsilon=1e-5))
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=0.1, atol=0.1)
