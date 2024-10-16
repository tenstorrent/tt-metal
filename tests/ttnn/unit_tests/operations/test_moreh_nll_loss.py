# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_torch,
    to_ttnn,
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

    tt_input = to_ttnn(torch_input, device=device)
    tt_target = to_ttnn(torch_target, device=device, dtype=npu_index_dtype)
    tt_weight = to_ttnn(torch_weight, device=device)
    tt_divisor = to_ttnn(torch_divisor, device=device)
    tt_output = to_ttnn(torch_output, device=device)

    return tt_input, tt_target, tt_weight, tt_divisor, tt_output


def run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)

    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )

    assert reduction in ["sum", "mean"]

    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        reduction,  # reduction_mean,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_tensor=tt_output,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )

    tt_loss = to_torch(tt_loss, shape=[1])
    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")
    assert passing


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
    reduction = "mean"
    if reduction_mean == False:
        tt_divisor = None
        reduction = "sum"
    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        reduction,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_tensor=tt_output,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )

    # run backward
    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    tt_output_grad = to_ttnn(output_grad, device=device)
    tt_input_grad = to_ttnn(torch_input, device=device)

    tt_input_grad = ttnn.operations.moreh.nll_loss_backward(
        target_tensor=tt_target,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_grad_tensor=tt_output_grad,
        input_grad_tensor=tt_input_grad,
        ignore_index=ignore_index,
        reduction_mean=reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )
    tt_input_grad = to_torch(tt_input_grad, shape=shape)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [3000, 100],
        [200, 100, 90],
        [5, 50, 2, 7, 50, 70],
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("none_weight", [True, False])
def test_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device):
    torch.manual_seed(0)

    run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device)


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [5, 6, 7],
        [5, 6, 8, 9],
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_moreh_nll_loss_callback(shape, reduction, device, use_program_cache):
    torch.manual_seed(0)
    ignore_index = 0

    num_program_cache_entries_list = []
    for i in range(4):
        if i < 2:
            none_weight = True
        else:
            none_weight = False

        run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)

        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert (
        num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
        and num_program_cache_entries_list[2] == num_program_cache_entries_list[3]
    )


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [10, 20, 30],
        [10, 20, 30, 40],
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_nll_loss_compute_kernel_options(
    shape, ignore_index, reduction, none_weight, compute_kernel_options, device
):
    torch.manual_seed(0)

    run_moreh_nll_loss(
        shape, ignore_index, reduction, none_weight, device, compute_kernel_options=compute_kernel_options
    )


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
def test_moreh_nll_loss_backward_test_callback(shape, reduction_mean, device, use_program_cache):
    torch.manual_seed(0)

    ignore_index = 0

    num_program_cache_entries_list = []
    for i in range(4):
        if i < 2:
            none_weight = True
        else:
            none_weight = False

        run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)

        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert (
        num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
        and num_program_cache_entries_list[2] == num_program_cache_entries_list[3]
    )


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
