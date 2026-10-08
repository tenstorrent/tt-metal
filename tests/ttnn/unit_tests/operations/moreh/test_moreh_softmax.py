# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_logsoftmax import (
    run_moreh_logsoftmax_backward_test,
    run_moreh_logsoftmax_test,
)
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_softmax import (
    run_moreh_softmax_backward_test,
    run_moreh_softmax_test,
)
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_softmin import (
    run_moreh_softmin_backward_test,
    run_moreh_softmin_test,
)
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

Strategy = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy
BackwardStrategy = ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy

# Each factory with the strategy that forces it. Every shape spans several tiles along dim, so the large factories
# stream more than one pass.
FACTORIES = [
    ([2, 32, 128], 2, Strategy.SMALL_W, BackwardStrategy.SMALL_W),
    ([2, 32, 128], 2, Strategy.LARGE_W, BackwardStrategy.LARGE_W),
    ([2, 128, 32], 1, Strategy.SMALL_H, BackwardStrategy.SMALL_H),
    ([2, 128, 32], 1, Strategy.LARGE_H, BackwardStrategy.LARGE_H),
    ([15, 32, 32], 0, Strategy.LARGE_C, BackwardStrategy.LARGE_C),
]
FACTORY_IDS = ["w_small", "w_large", "h_small", "h_large", "c_large"]

# The compute kernels pick the op with defines: SOFTMAX or SOFTMIN, plus LOG for logsoftmax.
# (forward helper, backward helper, forward tolerance, backward tolerance)
OPS = [
    (run_moreh_softmax_test, run_moreh_softmax_backward_test, 0.05, 0.05),
    (run_moreh_softmin_test, run_moreh_softmin_backward_test, 0.05, 0.05),
    (run_moreh_logsoftmax_test, run_moreh_logsoftmax_backward_test, 0.1, 0.5),
]
OP_IDS = ["softmax", "softmin", "logsoftmax"]

# The reduced dim spans three tiles, the last one partly filled, so the readers mask it.
UNALIGNED = [
    ([1, 1, 10, 74], 3, Strategy.SMALL_W, BackwardStrategy.SMALL_W),
    ([1, 1, 10, 74], 3, Strategy.LARGE_W, BackwardStrategy.LARGE_W),
    ([1, 1, 74, 32], 2, Strategy.SMALL_H, BackwardStrategy.SMALL_H),
    ([1, 1, 74, 32], 2, Strategy.LARGE_H, BackwardStrategy.LARGE_H),
]
UNALIGNED_IDS = ["w_small", "w_large", "h_small", "h_large"]


# Not the nightly helpers: they pad tiles with 0, which an unmasked softmax sum or backward reduce absorbs, so a
# broken mask would pass. NaN padding makes it fail. They also can't pass a provided output to the backward op.
def run_moreh_softmax_nan_pad_test(shape, dim, strategy, device, provide_output=False):
    torch_input = torch.randint(0, 4, shape).to(torch.bfloat16) + 100
    torch_output = torch.softmax(torch_input, dim)

    # NaN, so an output the op never writes fails: every softmax value here is within the tolerance of zero.
    tt_output = (
        create_ttnn_tilized_tensor(torch.full(shape, float("nan")), device, ttnn.bfloat16) if provide_output else None
    )
    result = ttnn.moreh_softmax(
        create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16),
        dim,
        output_tensor=tt_output,
        strategy=strategy,
        compute_kernel_config=get_compute_kernel_options(False),
    )
    # With a provided output, check that buffer itself: the op must write into it.
    actual = ttnn.to_torch(tt_output if provide_output else result)

    passing, output_pcc = comp_allclose_and_pcc(torch_output, actual, rtol=0.05, atol=0.05)
    assert passing, output_pcc


def run_moreh_softmax_backward_nan_pad_test(shape, dim, strategy, device, provide_output=False):
    torch_input = torch.randint(0, 4, shape).to(torch.bfloat16).requires_grad_(True)
    torch_output_grad = torch.randint(0, 4, shape).to(torch.bfloat16)
    torch_output = torch.softmax(torch_input, dim)
    torch_output.backward(torch_output_grad)

    # NaN, so an input_grad the op never writes fails.
    tt_input_grad = (
        create_ttnn_tilized_tensor(torch.full(shape, float("nan")), device, ttnn.bfloat16) if provide_output else None
    )
    result = ttnn.moreh_softmax_backward(
        create_ttnn_tilized_tensor(torch_output.detach(), device, ttnn.bfloat16),
        create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16),
        dim,
        input_grad_tensor=tt_input_grad,
        strategy=strategy,
        compute_kernel_config=get_compute_kernel_options(False),
    )
    # With a provided input_grad, check that buffer itself: the op must write into it.
    actual = ttnn.to_torch(tt_input_grad if provide_output else result)

    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, actual, rtol=0.05, atol=0.05)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize("op", OPS, ids=OP_IDS)
@pytest.mark.parametrize("factory", FACTORIES, ids=FACTORY_IDS)
def test_moreh_softmax(factory, op, device):
    torch.manual_seed(0)
    shape, dim, strategy, _ = factory
    run_test, _, tol, _ = op
    run_test(
        shape,
        dim,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        tol,
        tol,
        True,
        strategy=strategy,
        compute_kernel_options=False,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("op", OPS, ids=OP_IDS)
@pytest.mark.parametrize("factory", FACTORIES, ids=FACTORY_IDS)
def test_moreh_softmax_backward(factory, op, device):
    torch.manual_seed(0)
    shape, dim, _, strategy = factory
    _, run_test, _, tol = op
    run_test(
        shape,
        dim,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        tol,
        tol,
        True,
        strategy=strategy,
        compute_kernel_options=False,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("factory", FACTORIES, ids=FACTORY_IDS)
def test_moreh_softmax_fp32_dest_acc(factory, device):
    torch.manual_seed(0)
    shape, dim, strategy, _ = factory
    run_moreh_softmax_test(
        shape,
        dim,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        0.05,
        0.05,
        True,
        strategy=strategy,
        compute_kernel_options=True,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("factory", FACTORIES, ids=FACTORY_IDS)
def test_moreh_softmax_backward_fp32_dest_acc(factory, device):
    torch.manual_seed(0)
    shape, dim, _, strategy = factory
    run_moreh_softmax_backward_test(
        shape,
        dim,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        0.05,
        0.05,
        True,
        strategy=strategy,
        compute_kernel_options=True,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("unaligned", UNALIGNED, ids=UNALIGNED_IDS)
def test_moreh_softmax_unaligned(unaligned, device):
    torch.manual_seed(0)
    shape, dim, strategy, _ = unaligned
    run_moreh_softmax_nan_pad_test(shape, dim, strategy, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize("unaligned", UNALIGNED, ids=UNALIGNED_IDS)
def test_moreh_softmax_backward_unaligned(unaligned, device):
    torch.manual_seed(0)
    shape, dim, _, strategy = unaligned
    run_moreh_softmax_backward_nan_pad_test(shape, dim, strategy, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape",
    [
        [2, 32, 128],
        # 64 tiles in a row: past the small factory's 512 KB circular-buffer budget, so the op picks the large one.
        [1, 1, 32, 2048],
    ],
    ids=["picks_w_small", "picks_w_large"],
)
def test_moreh_softmax_auto_strategy(shape, device):
    torch.manual_seed(0)
    # No strategy: the op picks the factory from dim and size.
    run_moreh_softmax_test(
        shape, len(shape) - 1, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, 0.05, 0.05, True, compute_kernel_options=False
    )


@pytest.mark.merge_gate
def test_moreh_softmax_provided_output(device):
    torch.manual_seed(0)
    run_moreh_softmax_nan_pad_test([2, 32, 128], 2, Strategy.SMALL_W, device, provide_output=True)


@pytest.mark.merge_gate
def test_moreh_softmax_backward_provided_output(device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_nan_pad_test([2, 32, 128], 2, BackwardStrategy.SMALL_W, device, provide_output=True)
