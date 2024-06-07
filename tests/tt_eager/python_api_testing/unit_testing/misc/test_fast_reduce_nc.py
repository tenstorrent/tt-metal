# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn.experimental as ttl
import ttnn
from models.utility_functions import comp_allclose_and_pcc, comp_pcc
from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device, *, with_padding=True, use_randint=True, dataformat=ttnn.bfloat16):
    npu_dtype = dataformat
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    if use_randint:
        torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    else:
        torch_input = torch.rand(input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.rand(output_shape, dtype=cpu_dtype)

    if with_padding:
        tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).to(npu_layout).to(device)
        tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 8, 128, 4096],
        [1, 8, 1024, 4096],
        [1, 8, 2048, 4096],
    ),
    ids=[
        "mixtral_128",
        "mixtral_1k",
        "mixtral_2k",
    ],
)
@pytest.mark.parametrize(
    "dims",
    ([1],),
    ids=[
        "1",
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("dataformat", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bfloat16", "bfloat8_b"])
def test_fast_reduce_nc(input_shape, dims, compute_kernel_options, dataformat, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device, dataformat=dataformat)

    torch_output = torch.sum(torch_input, dims, True)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output = ttl.tensor.fast_reduce_nc(tt_input, dims=dims, output=None, compute_kernel_config=compute_kernel_config)
    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # test for equivalance
    rtol = atol = 0.12
    if dataformat == ttnn.bfloat8_b:
        passing, output_pcc = comp_pcc(torch_output, tt_output_cpu, pcc=0.999)
    else:
        passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing
