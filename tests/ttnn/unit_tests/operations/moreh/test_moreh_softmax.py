# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device

SoftmaxStrategy = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy
SoftmaxBackwardStrategy = ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy

INPUT_SHAPE = [2, TILE_HEIGHT, TILE_WIDTH]
# The nightly not-multiple-of-32 shapes: the reduced dim spans three tiles, the last one masked.
W_UNALIGNED_SHAPE = [1, 1, 10, TILE_WIDTH * 2 + 10]
H_UNALIGNED_SHAPE = [1, 1, TILE_HEIGHT * 2 + 10, TILE_WIDTH]

# On a rank-3 input: dim 2 is W, dim 1 is H, dim 0 is C (only the large C factory exists).
FACTORY_IDS = ["w_small", "w_large", "h_small", "h_large", "c_large"]

CORNER_CASE_IDS = [
    "w_large_multi_tile",
    "h_large_multi_tile",
    "c_large_multi_tile",
    "w_small_unaligned",
    "w_large_unaligned",
    "h_small_unaligned",
    "h_large_unaligned",
    "c_rank_4",
    "fp32_acc",
]


def run_moreh_softmax_test(
    ttnn_op,
    torch_op,
    input_shape,
    dim,
    tol,
    device,
    strategy=SoftmaxStrategy.NONE,
    fp32_dest_acc_en=False,
    provide_output=False,
):
    torch_input = torch.randint(0, 4, input_shape).to(torch.bfloat16) + 100
    torch_output = torch_op(torch_input, dim)

    tt_input = create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16)
    tt_output = create_ttnn_tilized_tensor(torch.zeros(input_shape), device, ttnn.bfloat16) if provide_output else None
    tt_output = ttnn.to_torch(
        ttnn_op(
            tt_input,
            dim,
            output_tensor=tt_output,
            strategy=strategy,
            compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
        )
    )

    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output, rtol=tol, atol=tol)
    assert passing, output_pcc


def run_moreh_softmax_backward_test(
    ttnn_op,
    torch_op,
    input_shape,
    dim,
    tol,
    device,
    strategy=SoftmaxBackwardStrategy.NONE,
    fp32_dest_acc_en=False,
    provide_output=False,
):
    torch_input = torch.randint(0, 4, input_shape).to(torch.bfloat16).requires_grad_(True)
    torch_output_grad = torch.randint(0, 4, input_shape).to(torch.bfloat16)
    torch_output = torch_op(torch_input, dim)
    torch_output.backward(torch_output_grad)

    tt_output = create_ttnn_tilized_tensor(torch_output.detach(), device, ttnn.bfloat16)
    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input_grad = (
        create_ttnn_tilized_tensor(torch.zeros(input_shape), device, ttnn.bfloat16) if provide_output else None
    )
    tt_input_grad = ttnn.to_torch(
        ttnn_op(
            tt_output,
            tt_output_grad,
            dim,
            input_grad_tensor=tt_input_grad,
            strategy=strategy,
            compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
        )
    )

    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad, rtol=tol, atol=tol)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "dim, strategy",
    [
        (2, SoftmaxStrategy.SMALL_W),
        (2, SoftmaxStrategy.LARGE_W),
        (1, SoftmaxStrategy.SMALL_H),
        (1, SoftmaxStrategy.LARGE_H),
        (0, SoftmaxStrategy.LARGE_C),
    ],
    ids=FACTORY_IDS,
)
def test_moreh_softmax(dim, strategy, device):
    torch.manual_seed(0)
    run_moreh_softmax_test(ttnn.moreh_softmax, torch.softmax, INPUT_SHAPE, dim, 0.05, device, strategy)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, strategy, fp32_dest_acc_en",
    [
        ([2, TILE_HEIGHT, TILE_WIDTH * 4], 2, SoftmaxStrategy.LARGE_W, False),
        ([2, TILE_HEIGHT * 4, TILE_WIDTH], 1, SoftmaxStrategy.LARGE_H, False),
        ([15, TILE_HEIGHT, TILE_WIDTH], 0, SoftmaxStrategy.LARGE_C, False),
        (W_UNALIGNED_SHAPE, 3, SoftmaxStrategy.SMALL_W, False),
        (W_UNALIGNED_SHAPE, 3, SoftmaxStrategy.LARGE_W, False),
        (H_UNALIGNED_SHAPE, 2, SoftmaxStrategy.SMALL_H, False),
        (H_UNALIGNED_SHAPE, 2, SoftmaxStrategy.LARGE_H, False),
        ([2, 3, TILE_HEIGHT * 2, TILE_WIDTH], 1, SoftmaxStrategy.LARGE_C, False),
        (W_UNALIGNED_SHAPE, 3, SoftmaxStrategy.NONE, True),
    ],
    ids=CORNER_CASE_IDS,
)
def test_moreh_softmax_corner_cases(input_shape, dim, strategy, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_softmax_test(
        ttnn.moreh_softmax, torch.softmax, input_shape, dim, 0.05, device, strategy, fp32_dest_acc_en=fp32_dest_acc_en
    )


@pytest.mark.merge_gate
def test_moreh_softmax_provided_output(device):
    torch.manual_seed(0)
    run_moreh_softmax_test(ttnn.moreh_softmax, torch.softmax, INPUT_SHAPE, 2, 0.05, device, provide_output=True)


@pytest.mark.merge_gate
def test_moreh_softmax_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_softmax_test(ttnn.moreh_softmax, torch.softmax, INPUT_SHAPE, 2, 0.05, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_softmax_test(ttnn.moreh_softmax, torch.softmax, INPUT_SHAPE, 2, 0.05, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "dim, strategy",
    [
        (2, SoftmaxBackwardStrategy.SMALL_W),
        (2, SoftmaxBackwardStrategy.LARGE_W),
        (1, SoftmaxBackwardStrategy.SMALL_H),
        (1, SoftmaxBackwardStrategy.LARGE_H),
        (0, SoftmaxBackwardStrategy.LARGE_C),
    ],
    ids=FACTORY_IDS,
)
def test_moreh_softmax_backward(dim, strategy, device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_softmax_backward, torch.softmax, INPUT_SHAPE, dim, 0.05, device, strategy
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, strategy, fp32_dest_acc_en",
    [
        ([2, TILE_HEIGHT, TILE_WIDTH * 4], 2, SoftmaxBackwardStrategy.LARGE_W, False),
        ([2, TILE_HEIGHT * 4, TILE_WIDTH], 1, SoftmaxBackwardStrategy.LARGE_H, False),
        ([15, TILE_HEIGHT, TILE_WIDTH], 0, SoftmaxBackwardStrategy.LARGE_C, False),
        (W_UNALIGNED_SHAPE, 3, SoftmaxBackwardStrategy.SMALL_W, False),
        (W_UNALIGNED_SHAPE, 3, SoftmaxBackwardStrategy.LARGE_W, False),
        (H_UNALIGNED_SHAPE, 2, SoftmaxBackwardStrategy.SMALL_H, False),
        (H_UNALIGNED_SHAPE, 2, SoftmaxBackwardStrategy.LARGE_H, False),
        ([2, 3, TILE_HEIGHT * 2, TILE_WIDTH], 1, SoftmaxBackwardStrategy.LARGE_C, False),
        (W_UNALIGNED_SHAPE, 3, SoftmaxBackwardStrategy.NONE, True),
    ],
    ids=CORNER_CASE_IDS,
)
def test_moreh_softmax_backward_corner_cases(input_shape, dim, strategy, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_softmax_backward,
        torch.softmax,
        input_shape,
        dim,
        0.05,
        device,
        strategy,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )


@pytest.mark.merge_gate
def test_moreh_softmax_backward_provided_output(device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_softmax_backward, torch.softmax, INPUT_SHAPE, 2, 0.05, device, provide_output=True
    )


@pytest.mark.merge_gate
def test_moreh_softmax_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_softmax_backward_test(ttnn.moreh_softmax_backward, torch.softmax, INPUT_SHAPE, 2, 0.05, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_softmax_backward_test(ttnn.moreh_softmax_backward, torch.softmax, INPUT_SHAPE, 2, 0.05, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_logsoftmax(device):
    torch.manual_seed(0)
    run_moreh_softmax_test(ttnn.moreh_logsoftmax, torch.nn.functional.log_softmax, INPUT_SHAPE, 2, 0.1, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize("input_shape, dim", [(W_UNALIGNED_SHAPE, 3)], ids=["w_unaligned"])
def test_moreh_logsoftmax_corner_cases(input_shape, dim, device):
    torch.manual_seed(0)
    run_moreh_softmax_test(ttnn.moreh_logsoftmax, torch.nn.functional.log_softmax, input_shape, dim, 0.1, device)


@pytest.mark.merge_gate
def test_moreh_logsoftmax_backward(device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_logsoftmax_backward, torch.nn.functional.log_softmax, INPUT_SHAPE, 2, 0.5, device
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("input_shape, dim", [(W_UNALIGNED_SHAPE, 3)], ids=["w_unaligned"])
def test_moreh_logsoftmax_backward_corner_cases(input_shape, dim, device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_logsoftmax_backward, torch.nn.functional.log_softmax, input_shape, dim, 0.5, device
    )


@pytest.mark.merge_gate
def test_moreh_softmin(device):
    torch.manual_seed(0)
    run_moreh_softmax_test(ttnn.moreh_softmin, torch.nn.functional.softmin, INPUT_SHAPE, 2, 0.05, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize("input_shape, dim", [(H_UNALIGNED_SHAPE, 2)], ids=["h_unaligned"])
def test_moreh_softmin_corner_cases(input_shape, dim, device):
    torch.manual_seed(0)
    run_moreh_softmax_test(ttnn.moreh_softmin, torch.nn.functional.softmin, input_shape, dim, 0.05, device)


@pytest.mark.merge_gate
def test_moreh_softmin_backward(device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_softmin_backward, torch.nn.functional.softmin, INPUT_SHAPE, 2, 0.05, device
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("input_shape, dim", [(H_UNALIGNED_SHAPE, 2)], ids=["h_unaligned"])
def test_moreh_softmin_backward_corner_cases(input_shape, dim, device):
    torch.manual_seed(0)
    run_moreh_softmax_backward_test(
        ttnn.moreh_softmin_backward, torch.nn.functional.softmin, input_shape, dim, 0.05, device
    )
