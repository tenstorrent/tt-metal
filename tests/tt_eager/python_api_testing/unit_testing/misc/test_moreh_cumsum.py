# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype, requires_grad=True)
    torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)

    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


def get_backward_tensors(output_grad_shape, input_grad_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype, requires_grad=True)
    torch_input_grad = torch.randint(-2, 3, input_grad_shape, dtype=cpu_dtype)

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
    "dim",
    (
        0,
        1,
    ),
    ids=["0", "1"],
)
def test_moreh_cumsum_dim(input_shape, dim, device):
    output_shape = input_shape.copy()

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device)

    torch_output = torch.cumsum(torch_input, dim)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.moreh_cumsum(tt_input, tt_output, dim=dim)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

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
    "dim",
    (
        0,
        1,
    ),
    ids=["0", "1"],
)
def test_moreh_cumsumsum_backward(input_shape, dim, device):
    output_shape = input_shape.copy()

    (_, _, torch_input) = get_tensors(input_shape, output_shape, device)
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(output_shape, input_shape, device)

    torch_output = torch.cumsum(torch_input, dim)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_cumsum_backward(tt_output_grad, tt_input_grad, dim=dim)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(input_shape)
        .to_torch()
    )

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing
