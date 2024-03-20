# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc, skip_for_wormhole_b0

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, optional_tensor, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype, requires_grad=True)
    torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)

    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    if optional_tensor == True:
        optional_torch = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
        optional_output = (
            ttl.tensor.Tensor(optional_torch, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        )
        return tt_input, tt_output, torch_input, optional_output, optional_torch
    else:
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


# Dongjin : WH_B0 skips this test due to the problem of sum reduction for w-dim.
@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "input_shape",
    (
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1]),
        ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1",
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
    "optional_tensor",
    (
        True,
        False,
    ),
    ids=["True", "False"],
)
def test_moreh_sum_dims(input_shape, dims, optional_tensor, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    if optional_tensor:
        (tt_input, tt_output, torch_input, optional_output, optional_torch) = get_tensors(
            input_shape, output_shape, optional_tensor, device
        )

        tt_output_cpu = ttl.operations.primary.moreh_sum(tt_input, tt_output, dims=dims, output_tensor=optional_output)

    else:
        (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, optional_tensor, device)

        tt_output_cpu = ttl.operations.primary.moreh_sum(tt_input, tt_output, dims=dims)

    torch_output = torch.sum(torch_input, dims, True)
    tt_out_tensor = tt_output_cpu[0].cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    if optional_tensor:
        tt_optional_out_tensor = tt_output_cpu[1].cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
        rtol = atol = 0.12

        passing, output_pcc = comp_allclose_and_pcc(
            optional_torch, tt_optional_out_tensor, pcc=0.999, rtol=rtol, atol=atol
        )

        logger.debug(f"Optional Out passing={passing}")
        logger.debug(f"optional Output pcc={output_pcc}")

        assert passing

    rtol = atol = 0.12
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_out_tensor, pcc=0.999, rtol=rtol, atol=atol)

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
def test_moreh_sum_backward(input_shape, dims, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (_, _, torch_input) = get_tensors(input_shape, output_shape, device)
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(output_shape, input_shape, device)

    torch_output = torch.sum(torch_input, dims, True)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_tensor = None
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_sum_backward(tt_output_grad, tt_input_grad, tt_output_tensor)
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
