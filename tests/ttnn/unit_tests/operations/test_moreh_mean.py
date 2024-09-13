# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
import ttnn
from models.utility_functions import comp_allclose

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
    TILE_HEIGHT,
    TILE_WIDTH,
    check_dim,
)


def get_torch_tensors(input_shape, use_randint=False):
    cpu_dtype = torch.bfloat16

    if use_randint:
        torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
    else:
        torch_input = torch.rand(input_shape, dtype=cpu_dtype)

    torch_input.requires_grad_()
    return torch_input


def get_tt_tensors(torch_input, output_shape, device):
    torch_input = torch_input.bfloat16()
    torch_output = torch.empty(output_shape, dtype=torch.bfloat16)

    tt_input = to_npu(torch_input, device)
    tt_output = to_npu(torch_output, device)
    return tt_input, tt_output


def get_torch_backward_tensors(output_grad_shape, use_randint=False):
    cpu_dtype = torch.bfloat16

    if use_randint:
        torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype)
    else:
        torch_output_grad = torch.rand(output_grad_shape, dtype=cpu_dtype)

    return torch_output_grad


def get_tt_backward_tensors(torch_output_grad, input_grad_shape, device):
    cpu_dtype = torch.bfloat16

    torch_input_grad = torch.empty(input_grad_shape, dtype=cpu_dtype)

    tt_input_grad = to_npu(torch_input_grad, device)
    tt_output_grad = to_npu(torch_output_grad, device)

    return tt_output_grad, tt_input_grad


def run_moreh_mean(input_shape_dim, device, keepdim=False, compute_kernel_options=None):
    input_shape, dim = input_shape_dim

    check_dim(input_shape, dim, keepdim)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # run torch
    torch_input = get_torch_tensors(input_shape)
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)

    # run tt
    (tt_input, tt_output) = get_tt_tensors(torch_input, torch_output.shape, device)

    ttnn.operations.moreh.mean(
        tt_input, dim=dim, keepdim=keepdim, output=tt_output, compute_kernel_config=compute_kernel_config
    )
    tt_output_cpu = to_cpu(tt_output, torch_output.shape)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose(torch_output, tt_output_cpu, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


def run_moreh_mean_backward(input_shape_dim, device, keepdim=False, compute_kernel_options=None, create_output=False):
    input_shape, dim = input_shape_dim

    check_dim(input_shape, dim, keepdim)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # run torch
    torch_input = get_torch_tensors(input_shape)
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)

    torch_output_grad = get_torch_backward_tensors(torch_output.shape)

    torch_output.backward(torch_output_grad)

    # run_tt
    tt_output_grad, tt_input_grad = get_tt_backward_tensors(torch_output_grad, torch_input.shape, device)

    if create_output:
        input_grad_shape = ttnn._ttnn.types.Shape(torch_input.shape)
        tt_input_grad = ttnn.operations.moreh.mean_backward(
            tt_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad_shape=input_grad_shape,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        ttnn.operations.moreh.mean_backward(
            tt_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad=tt_input_grad,
            compute_kernel_config=compute_kernel_config,
        )

    tt_input_grad_cpu = to_cpu(tt_input_grad, torch_input.grad.shape)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose(torch_input.grad, tt_input_grad_cpu, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw single tile
        [[TILE_HEIGHT - 15, TILE_WIDTH - 10], [0]],  # h
        [[TILE_HEIGHT - 15, TILE_WIDTH - 10], [1]],  # w
        # hw multiple tiles
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [0]],  # h
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [1]],  # w
        # ncd single tile
        [[3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [0]],  # n
        [[3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [1]],  # c
        [[3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [2]],  # d
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0]],  # n
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1]],  # c
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [2]],  # d
        # multiple dims
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0, 1, 2]],
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1, 2]],
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0, 2, 4]],
        # all dims
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], None],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_mean(input_shape_dim, keepdim, device):
    torch.manual_seed(2023)

    run_moreh_mean(input_shape_dim, device, keepdim)


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw multiple tiles
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [0]],  # h
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [1]],  # w
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1]],  # c
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_mean_compute_kernel_options(input_shape_dim, compute_kernel_options, device):
    torch.manual_seed(2023)

    run_moreh_mean(input_shape_dim, device, keepdim=True, compute_kernel_options=compute_kernel_options)


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw multiple tiles
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [0]],  # h
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [1]],  # w
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1]],  # c
    ],
)
def test_moreh_mean_callback(input_shape_dim, device, use_program_cache):
    torch.manual_seed(2023)

    for _ in range(2):
        run_moreh_mean(input_shape_dim, device, keepdim=True)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw single tile
        [[TILE_HEIGHT - 15, TILE_WIDTH - 10], [1]],  # h
        [[TILE_HEIGHT - 15, TILE_WIDTH - 10], [1]],  # w
        # hw multiple tiles
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [0]],  # h
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [1]],  # w
        # # ncd single tile
        [[3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [0]],  # n
        [[3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [1]],  # c
        [[3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [2]],  # d
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0]],  # n
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1]],  # c
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [2]],  # d
        # multiple dims
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0, 1, 2]],
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1, 2]],
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0, 2, 4]],
        # all dims
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], None],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_mean_backward(input_shape_dim, keepdim, device):
    torch.manual_seed(2023)

    run_moreh_mean_backward(input_shape_dim, device, keepdim=keepdim)


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw multiple tiles
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [0]],  # h
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [1]],  # w
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1]],  # c
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_mean_backward_compute_kernel_options(input_shape_dim, compute_kernel_options, device):
    torch.manual_seed(2023)

    run_moreh_mean_backward(input_shape_dim, device, keepdim=True, compute_kernel_options=compute_kernel_options)


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw multiple tiles
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [0]],  # h
        [[TILE_HEIGHT * 4, TILE_WIDTH * 5], [1]],  # w
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [1]],  # c
    ],
)
def test_moreh_mean_backward_callback(input_shape_dim, device, use_program_cache):
    torch.manual_seed(2023)

    for _ in range(2):
        run_moreh_mean_backward(input_shape_dim, device, keepdim=True)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)


@pytest.mark.parametrize(
    "input_shape_dim",
    [
        # hw multiple tiles
        [[TILE_HEIGHT * 4 - 10, TILE_WIDTH * 5 - 20], [0]],  # h
        [[TILE_HEIGHT * 4 - 10, TILE_WIDTH * 5 - 20], [1]],  # h
        # ncd multiple tile
        [[3, 4, 5, TILE_HEIGHT * 3 - 15, TILE_WIDTH * 4 - 10], [0, 2]],  # c
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_mean_backward_create_output(input_shape_dim, keepdim, device):
    torch.manual_seed(2023)

    run_moreh_mean_backward(input_shape_dim, device, keepdim=keepdim, create_output=True)
