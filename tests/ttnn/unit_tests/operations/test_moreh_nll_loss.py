# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
)


def get_torch_tensors(shape):
    C = shape[1]
    target_shape = shape[:1] + shape[2:]

    cpu_dtype = torch.float32
    cpu_index_dtype = torch.long

    torch_input = torch.rand(shape, dtype=cpu_dtype).requires_grad_()
    torch_target = torch.randint(0, C, target_shape, dtype=cpu_index_dtype)
    torch_weight = torch.rand(C, dtype=cpu_dtype)
    torch_divisor = torch.tensor([0], dtype=cpu_dtype)
    torch_output = torch.tensor([0], dtype=cpu_dtype)

    return torch_input, torch_target, torch_weight, torch_divisor, torch_output


def get_tt_tensors(torch_input, torch_target, torch_weight, torch_divisor, torch_output, device):
    npu_index_dtype = ttnn.int32

    tt_input = to_npu(torch_input, device)
    tt_target = to_npu(torch_target, device, npu_dtype=npu_index_dtype)
    tt_weight = to_npu(torch_weight, device)
    tt_divisor = to_npu(torch_divisor, device)
    tt_output = to_npu(torch_output, device)

    return tt_input, tt_target, tt_weight, tt_divisor, tt_output


# def run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device, compute_kernel_options=None):
#     compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

#     (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)

#     if none_weight:
#         torch_weight = None

#     nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
#     torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

#     (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
#         torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
#     )

#     reduction_mean = reduction == "mean"
#     tt_loss = ttnn.experimental.operations.primary.moreh_nll_loss(
#         tt_input,
#         tt_target,
#         tt_weight,
#         tt_divisor,
#         tt_output,
#         ignore_index,
#         reduction_mean,
#         compute_kernel_config=compute_kernel_config,
#     )

#     tt_loss_to_cpu = to_cpu(tt_loss, [1])
#     rtol = atol = 0.05
#     passing, out = comp_allclose_and_pcc(torch_loss, tt_loss_to_cpu, pcc=0.999, rtol=rtol, atol=atol)
#     logger.debug(f"Out passing (param)={passing}")
#     logger.debug(f"Output pcc={out}")
#     assert passing


def run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)
    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )
    if reduction_mean == False:
        tt_divisor = None
    tt_loss = ttnn.experimental.operations.primary.moreh_nll_loss(
        tt_input,
        tt_target,
        tt_weight,
        tt_divisor,
        tt_output,
        ignore_index,
        reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )

    # run backward
    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    tt_output_grad = to_npu(output_grad, device)
    tt_input_grad = to_npu(torch_input, device)

    tt_input_grad = ttnn.experimental.operations.primary.moreh_nll_loss_backward(
        tt_target,
        tt_weight,
        tt_divisor,
        tt_output_grad,
        tt_input_grad,
        ignore_index,
        reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )
    tt_input_grad_to_cpu = to_cpu(tt_input_grad, shape)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_to_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


# @pytest.mark.parametrize(
#     "shape",
#     [[5, 10], [3000, 100], [200, 100, 90], [5, 50, 2, 7, 50, 70]],
# )
# @pytest.mark.parametrize("ignore_index", [1])
# @pytest.mark.parametrize("reduction", ["mean", "sum"])
# @pytest.mark.parametrize("none_weight", [True, False])
# def test_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device):
#     torch.manual_seed(0)

#     run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device)


# @pytest.mark.parametrize(
#     "shape",
#     [
#         [5, 10],
#         [5, 6, 7],
#         [5, 6, 8, 9],
#     ],
# )
# @pytest.mark.parametrize("reduction", ["mean", "sum"])
# @pytest.mark.parametrize("none_weight", [True, False])
# def test_moreh_nll_loss_callback(shape, reduction, none_weight, device, use_program_cache):
#     torch.manual_seed(0)

#     ignore_idx = 0

#     for _ in range(2):
#         run_moreh_nll_loss(shape, ignore_idx, reduction, none_weight, device)
#         torch_dummy = torch.randn([32, 32])
#         tt_dummy = to_npu(torch_dummy, device)


@pytest.mark.parametrize(
    "shape",
    [
        [400, 300],
        [20, 300, 320],
        [3, 4, 32 * 5, 32 * 6],
        [5, 2, 5, 40, 70],
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
def test_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device):
    torch.manual_seed(0)

    run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device)


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3],
        [2, 3, 4],
        [2, 3, 5, 4],
    ],
)
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
def test_moreh_nll_loss_backward_test_callback(shape, reduction_mean, none_weight, device, use_program_cache):
    torch.manual_seed(0)

    ignore_index = 0

    for _ in range(2):
        run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)


# @pytest.mark.parametrize(
#     "shape",
#     [
#         [5, 10],
#         [10, 20, 30],
#         [10, 20, 30, 40],
#     ],
# )
# @pytest.mark.parametrize("ignore_index", [1])
# @pytest.mark.parametrize("reduction", ["mean", "sum"])
# @pytest.mark.parametrize("none_weight", [True, False])
# @pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
# def test_moreh_nll_loss_compute_kernel_options(
#     shape, ignore_index, reduction, none_weight, compute_kernel_options, device
# ):
#     torch.manual_seed(0)

#     run_moreh_nll_loss(
#         shape, ignore_index, reduction, none_weight, device, compute_kernel_options=compute_kernel_options
#     )


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [10, 20, 30],
        [10, 20, 30, 40],
    ],
)
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_nll_loss_backward_compute_kernel_options(
    shape, reduction_mean, none_weight, compute_kernel_options, device
):
    torch.manual_seed(0)

    ignore_index = 0

    run_moreh_nll_loss_backward(
        shape, ignore_index, reduction_mean, none_weight, device, compute_kernel_options=compute_kernel_options
    )
