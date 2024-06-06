# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device, *, with_padding=True, use_randint=True):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
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
@pytest.mark.parametrize(
    "use_provide_output",
    (False,),
    ids=[
        "False",
    ],
)
def test_fast_reduce_nc(input_shape, dims, compute_kernel_options, use_provide_output, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device)

    if not use_provide_output:
        tt_output = None

    torch_output = torch.sum(torch_input, dims, True)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.tensor.fast_reduce_nc(tt_input, dims=dims, output=tt_output, compute_kernel_config=compute_kernel_config)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # test for equivalance
    # TODO(Dongjin) : check while changing rtol after enabling fp32_dest_acc_en
    rtol = atol = 0.12
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing
