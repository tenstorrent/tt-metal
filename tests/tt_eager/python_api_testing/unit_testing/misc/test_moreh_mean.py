# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc, comp_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, use_randint, device):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_dtype = torch.float

    if use_randint:
        torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
        torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    else:
        torch_input = torch.rand(input_shape, dtype=cpu_dtype) * 200 - 100
        torch_output = torch.rand(output_shape, dtype=cpu_dtype) * 200 - 100

    torch_input = torch_input.bfloat16().requires_grad_()
    torch_output = torch_output.bfloat16()
    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return tt_input, tt_output, torch_input


def get_backward_tensors(output_grad_shape, input_grad_shape, use_randint, device):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_dtype = torch.float

    if use_randint:
        torch_input_grad = torch.randint(-2, 3, input_grad_shape, dtype=cpu_dtype)
        torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype)
    else:
        torch_input_grad = torch.rand(input_grad_shape, dtype=cpu_dtype) * 200 - 100
        torch_output_grad = torch.rand(output_grad_shape, dtype=cpu_dtype) * 200 - 100

    torch_input_grad = torch_input_grad.bfloat16()
    torch_output_grad = torch_output_grad.bfloat16()
    tt_output_grad = ttl.tensor.Tensor(torch_output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_input_grad = ttl.tensor.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


@pytest.mark.parametrize(
    "input_shape",
    (
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 30 - 1]),
        ([4, 4, TILE_HEIGHT * 30 - 1, TILE_WIDTH * 12 - 1]),
        ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 30 - 1",
        "4, 4, TILE_HEIGHT * 30 - 1, TILE_WIDTH * 12 - 1",
        "8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 3],
        [2],
        [2, 3],
        [3],
    ),
    ids=["0", "0,1", "0,1,2", "0,1,2,3", "0,1,3", "0,2,3", "1", "1,2", "1,2,3", "1,3", "2", "2,3", "3"],
)
@pytest.mark.parametrize(
    "use_randint",
    (True, False),
    ids=["True", "False"],
)
def test_moreh_mean_dims(input_shape, dims, use_randint, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, use_randint, device)

    torch_output = torch.mean(torch_input, dims, True)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.moreh_mean(tt_input, tt_output, dims=dims)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # test for equivalance
    rtol = atol = 0.1
    if use_randint:
        passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    else:
        passing, output_pcc = comp_pcc(torch_output, tt_output_cpu, pcc=0.999)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "input_shape",
    (
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 30 - 1]),
        ([4, 4, TILE_HEIGHT * 30 - 1, TILE_WIDTH * 12 - 1]),
        ([8, 8, TILE_HEIGHT * 20 - 1, TILE_WIDTH * 20 - 1]),
    ),
    ids=[
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 30 - 1",
        "4, 4, TILE_HEIGHT * 30 - 1, TILE_WIDTH * 12 - 1",
        "8, 8, TILE_HEIGHT * 20 - 1, TILE_WIDTH * 20 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 3],
        [2],
        [2, 3],
        [3],
    ),
    ids=["0", "0,1", "0,1,2", "0,1,2,3", "0,1,3", "0,2,3", "1", "1,2", "1,2,3", "1,3", "2", "2,3", "3"],
)
@pytest.mark.parametrize(
    "use_randint",
    (True, False),
    ids=["True", "False"],
)
def test_moreh_mean_backward(input_shape, dims, use_randint, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (_, _, torch_input) = get_tensors(input_shape, output_shape, use_randint, device)
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(
        output_shape, input_shape, use_randint, device
    )

    torch_output = torch.mean(torch_input, dims, True)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_mean_backward(tt_output_grad, tt_input_grad)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(input_shape)
        .to_torch()
    )

    # test for equivalance
    rtol = atol = 0.1
    if use_randint:
        passing, output_pcc = comp_allclose_and_pcc(
            torch_input.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol
        )
    else:
        passing, output_pcc = comp_pcc(torch_input.grad, tt_input_grad_cpu, pcc=0.999)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing
