# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    check_dim,
    compute_kernel_ids,
    compute_kernel_options,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)


def run_moreh_mean(
    input_shape_dim,
    device,
    *,
    keepdim=False,
    compute_kernel_options=None,
    optional_output=False,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
):
    # TODO @mrshaw01: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")

    input_shape, dim = input_shape_dim
    check_dim(input_shape, dim, keepdim)

    # run torch
    torch_input = torch.rand(input_shape, dtype=torch_dtype)
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)

    # run ttnn
    ttnn_input = create_ttnn_tilized_tensor(torch_input, device, ttnn_dtype)
    ttnn_output = (
        create_ttnn_tilized_tensor(torch.empty_like(torch_output), device, ttnn_dtype) if optional_output else None
    )
    ttnn_output = ttnn.operations.moreh.mean(
        ttnn_input,
        dim=dim,
        keepdim=keepdim,
        output=ttnn_output,
        compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
    )
    output = ttnn.to_torch(ttnn_output)

    rtol = atol = 0.1
    passing, out = comp_allclose(torch_output, output, rtol=rtol, atol=atol)
    logger.info(f"passing={passing}")
    logger.info(f"out={out}")
    assert passing


def run_moreh_mean_backward(
    input_shape_dim,
    device,
    *,
    keepdim=False,
    compute_kernel_options=None,
    create_input_grad=False,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
):
    # TODO @mrshaw01: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")

    input_shape, dim = input_shape_dim
    check_dim(input_shape, dim, keepdim)

    # run torch
    torch_input = torch.rand(input_shape, dtype=torch_dtype)
    torch_input.requires_grad_()
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)
    torch_output_grad = torch.rand(torch_output.shape, dtype=torch_dtype)
    torch_output.backward(torch_output_grad)

    # run ttnn
    ttnn_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn_dtype)
    if create_input_grad:
        input_grad_shape = tuple(torch_input.shape)
        ttnn_input_grad = ttnn.operations.moreh.mean_backward(
            ttnn_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad_shape=input_grad_shape,
            compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
        )
    else:
        ttnn_input_grad = create_ttnn_tilized_tensor(torch.empty_like(torch_input), device, ttnn_dtype)
        ttnn.operations.moreh.mean_backward(
            ttnn_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad=ttnn_input_grad,
            compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
        )
    input_grad = ttnn.to_torch(ttnn_input_grad)

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose(torch_input.grad, input_grad, rtol=rtol, atol=atol)
    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")
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
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_moreh_mean_ttnn_dtype(input_shape_dim, keepdim, ttnn_dtype, device):
    torch.manual_seed(2024)
    run_moreh_mean(input_shape_dim, device, keepdim=keepdim, ttnn_dtype=ttnn_dtype)


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
    torch.manual_seed(2024)
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
@pytest.mark.parametrize("optional_output", [True, False])
def test_moreh_mean_optional_output(input_shape_dim, optional_output, device):
    torch.manual_seed(2024)
    run_moreh_mean(input_shape_dim, device, keepdim=True, optional_output=optional_output)


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
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_mean(input_shape_dim, device, keepdim=True)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


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
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_moreh_mean_backward_ttnn_dtype(ttnn_dtype, input_shape_dim, keepdim, device):
    torch.manual_seed(2024)
    run_moreh_mean_backward(input_shape_dim, device, keepdim=keepdim, ttnn_dtype=ttnn_dtype)


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
    torch.manual_seed(2024)
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
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_mean_backward(input_shape_dim, device, keepdim=True)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


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
@pytest.mark.parametrize("create_input_grad", [True, False])
def test_moreh_mean_backward_create_input_grad(input_shape_dim, keepdim, create_input_grad, device):
    torch.manual_seed(2024)
    run_moreh_mean_backward(input_shape_dim, device, keepdim=keepdim, create_input_grad=create_input_grad)
