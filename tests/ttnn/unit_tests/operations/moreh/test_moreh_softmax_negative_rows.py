# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Regression test: on a non-tile-aligned reduce dim, the MAX shift of moreh softmax / logsoftmax must
# ignore the padded lanes of the last tile. Padding masked to 0 wins the MAX whenever every real lane
# of a row is negative, so very negative rows (fully masked attention rows) underflow exp() to 0.

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

Strategy = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy

CASES = [
    ([1, 1, 32, 74], 3, Strategy.SMALL_W),
    ([1, 1, 32, 74], 3, Strategy.LARGE_W),
    ([1, 1, 74, 32], 2, Strategy.SMALL_H),
    ([1, 1, 74, 32], 2, Strategy.LARGE_H),
]
CASE_IDS = ["w_small", "w_large", "h_small", "h_large"]

# Aligned twins of CASES: same values, no padding in the reduce dim. These pass before the fix.
ALIGNED = [
    ([1, 1, 32, 64], 3, Strategy.SMALL_W),
    ([1, 1, 32, 64], 3, Strategy.LARGE_W),
    ([1, 1, 64, 32], 2, Strategy.SMALL_H),
    ([1, 1, 64, 32], 2, Strategy.LARGE_H),
]


def _negative_input(shape, fill):
    # Every real lane is negative; small jitter so rows are not perfectly uniform.
    return (torch.full(shape, fill) - torch.rand(shape)).to(torch.bfloat16)


def _run(op, torch_op, shape, dim, strategy, fill, device):
    torch_input = _negative_input(shape, fill)
    expected = torch_op(torch_input.float(), dim)
    result = op(
        create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16),
        dim,
        strategy=strategy,
        compute_kernel_config=get_compute_kernel_options(False),
    )
    actual = ttnn.to_torch(result).float()
    assert torch.isfinite(actual).all(), f"non-finite output: {actual.flatten()[:8]}"
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)


@pytest.mark.parametrize("fill", [-1e4, -200.0], ids=["masked_row", "very_negative"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_moreh_softmax_negative_rows_unaligned(case, fill, device):
    torch.manual_seed(0)
    shape, dim, strategy = case
    _run(ttnn.moreh_softmax, torch.softmax, shape, dim, strategy, fill, device)


@pytest.mark.parametrize("fill", [-1e4, -200.0], ids=["masked_row", "very_negative"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_moreh_logsoftmax_negative_rows_unaligned(case, fill, device):
    torch.manual_seed(0)
    shape, dim, strategy = case
    _run(ttnn.moreh_logsoftmax, torch.log_softmax, shape, dim, strategy, fill, device)


@pytest.mark.parametrize("case", ALIGNED, ids=CASE_IDS)
def test_moreh_softmax_negative_rows_aligned_control(case, device):
    torch.manual_seed(0)
    shape, dim, strategy = case
    _run(ttnn.moreh_softmax, torch.softmax, shape, dim, strategy, -1e4, device)
