# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc, is_wormhole_b0
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
    torch_output = torch.empty(target_shape, dtype=cpu_dtype)

    return torch_input, torch_target, torch_weight, torch_output


def get_tt_tensors(torch_input, torch_target, torch_weight, torch_output, device):
    npu_index_dtype = ttnn.int32

    tt_input = to_ttnn(torch_input, device=device)
    tt_target = to_ttnn(torch_target, device=device, dtype=npu_index_dtype)
    tt_weight = to_ttnn(torch_weight, device=device)
    tt_output = to_ttnn(torch_output, device=device)

    return tt_input, tt_target, tt_weight, tt_output


def get_tt_backward_tensors(torch_target, torch_weight, torch_output_grad, torch_input_grad, device):
    npu_index_dtype = ttnn.int32

    tt_target = to_ttnn(torch_target, device=device, dtype=npu_index_dtype)
    tt_weight = to_ttnn(torch_weight, device=device)
    tt_output_grad = to_ttnn(torch_output_grad, device=device)
    tt_input_grad = to_ttnn(torch_input_grad, device=device)

    return tt_target, tt_weight, tt_output_grad, tt_input_grad


def run_moreh_nll_loss_unreduced_backward(shape, ignore_index, none_weight, device, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # run torch
    (torch_input, torch_target, torch_weight, _) = get_torch_tensors(shape)
    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction="none")
    torch_loss = nll_loss(torch_input, torch_target)

    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    # run tt
    (tt_target, tt_weight, tt_output_grad, tt_input_grad) = get_tt_backward_tensors(
        torch_target, torch_weight, output_grad, torch_input.grad, device
    )

    tt_input_grad = ttnn.operations.moreh.nll_loss_unreduced_backward(
        tt_target,
        tt_output_grad,
        weight_tensor=tt_weight,
        input_grad_tensor=tt_input_grad,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )
    tt_input_grad = to_torch(tt_input_grad, shape=torch_input.grad.shape)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


def run_moreh_nll_loss_unreduced(shape, ignore_index, none_weight, device, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (torch_input, torch_target, torch_weight, torch_output) = get_torch_tensors(shape)

    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction="none")
    torch_loss = nll_loss(torch_input, torch_target)

    (tt_input, tt_target, tt_weight, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_output, device
    )

    reduction_mode = "none"

    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        reduction_mode,
        weight_tensor=tt_weight,
        divisor_tensor=None,
        output_tensor=tt_output,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )

    tt_loss = to_torch(tt_loss, shape=torch_target.shape)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize(
    "shape",
    [
        (5, 10),
        (500, 100),
        (4, 3, 2, 4, 50, 70),
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_nll_loss_unreduced(shape, ignore_index, none_weight, compute_kernel_options, device, use_program_cache):
    torch.manual_seed(0)

    run_moreh_nll_loss_unreduced(
        shape, ignore_index, none_weight, device, compute_kernel_options=compute_kernel_options
    )


@pytest.mark.parametrize(
    "shape",
    [
        (5, 10),
        (5, 10, 10),
        (5, 10, 10, 20),
    ],
)
def test_moreh_nll_loss_unreduced_callback(shape, device, use_program_cache):
    torch.manual_seed(0)

    ignore_index = 1
    num_program_cache_entries_list = []

    for i in range(4):
        if i < 2:
            none_weight = True
        else:
            none_weight = False

        run_moreh_nll_loss_unreduced(shape, ignore_index, none_weight, device)
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
        (32, 32),
        (400, 300),
        (20, 300, 320),
        (5, 2, 5, 40, 70),
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_nll_loss_unreduced_backward(
    shape, ignore_index, none_weight, compute_kernel_options, device, use_program_cache
):
    torch.manual_seed(0)

    run_moreh_nll_loss_unreduced_backward(
        shape, ignore_index, none_weight, device, compute_kernel_options=compute_kernel_options
    )


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4),
        (2, 3, 5, 4),
    ],
)
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("ignore_index", [0, -100])
def test_moreh_nll_loss_unreduced_backward_test_callback(shape, none_weight, device, ignore_index, use_program_cache):
    torch.manual_seed(0)

    num_program_cache_entries_list = []
    for i in range(4):
        if i < 2:
            none_weight = True
        else:
            none_weight = False

        run_moreh_nll_loss_unreduced_backward(shape, ignore_index, none_weight, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)

        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert (
        num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
        and num_program_cache_entries_list[2] == num_program_cache_entries_list[3]
    )
