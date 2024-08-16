# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(-100, 100, output_shape, dtype=cpu_dtype)

    tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        ([2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1]),
        ([9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1]),
        ([4, 3, TILE_HEIGHT * 3 - 1, TILE_WIDTH * 11 - 1]),
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1]),
        ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1",
        "9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1",
        "4, 3, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 11 - 1",
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1",
        "8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [
            0,
        ],
        [
            1,
        ],
    ),
    ids=["0", "1"],
)
# Support for dim 2,3 in composite_ops
def test_prod_dims(input_shape, dims, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device)

    torch_output = torch.prod(torch_input, dims[0], True)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = (
        ttnn.prod(tt_input, tt_output, dims=dims).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    )

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing
