# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)
from models.utility_functions import comp_allclose_and_pcc

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import TILE_HEIGHT, TILE_WIDTH


def get_tensors(input_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_input = torch.rand(input_shape, dtype=cpu_dtype, requires_grad=True)
    torch_output = torch.zeros(input_shape, dtype=cpu_dtype)

    tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


def get_backward_tensors(input_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_output_grad = torch.rand(input_shape, dtype=cpu_dtype)
    torch_input_grad = torch.zeros(input_shape, dtype=cpu_dtype)

    tt_output_grad = ttnn.Tensor(torch_output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_input_grad = ttnn.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


@pytest.mark.parametrize(
    "input_shape",
    (
        ([TILE_HEIGHT, TILE_WIDTH]),
        ([TILE_HEIGHT // 2, TILE_WIDTH // 2]),
        ([2, 3, 4, TILE_HEIGHT * 5 + TILE_HEIGHT // 2, TILE_WIDTH * 5 + TILE_WIDTH // 2]),
    ),
)
@pytest.mark.parametrize(
    "dim",
    (
        0,
        1,
        2,
        3,
        4,
    ),
    ids=[
        "0",
        "1",
        "2",
        "3",
        "4",
    ],
)
@pytest.mark.parametrize("use_provide_output", (True, False), ids=["True", "False"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_cumsum(input_shape, dim, use_provide_output, compute_kernel_options, device):
    input_rank = len(input_shape)
    if dim >= input_rank:
        pytest.skip(f"input dim {dim} exceeds the dims of input tensor {input_rank}")
    # TODO: remove this condition
    if dim == input_rank - 1:
        pytest.skip("not supported")

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, device)
    if not use_provide_output:
        tt_output = None

    torch_output = torch.cumsum(torch_input, dim)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = (
        ttnn.experimental.operations.primary.moreh_cumsum(
            tt_input, dim=dim, output=tt_output, compute_kernel_config=compute_kernel_config
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(input_shape)
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
        ([TILE_HEIGHT, TILE_WIDTH]),
        ([TILE_HEIGHT // 2, TILE_WIDTH // 2]),
        ([2 * TILE_HEIGHT - 7, 2 * TILE_WIDTH - 7]),
        ([TILE_HEIGHT + 6, TILE_WIDTH + 6]),
        ([2, 3, 4, TILE_HEIGHT * 5 + TILE_HEIGHT // 2, TILE_WIDTH * 5 + TILE_WIDTH // 2]),
    ),
)
@pytest.mark.parametrize(
    "dim",
    (
        0,
        1,
        2,
        3,
        4,
    ),
    ids=[
        "0",
        "1",
        "2",
        "3",
        "4",
    ],
)
@pytest.mark.parametrize("use_provide_output", (True, False), ids=["True", "False"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_cumsum_backward(input_shape, dim, use_provide_output, compute_kernel_options, device):
    input_rank = len(input_shape)
    if dim >= input_rank:
        pytest.skip(f"input dim {dim} exceeds the dims of input tensor {input_rank}")
    # TODO: remove this condition
    if dim == input_rank - 1:
        pytest.skip("last two dimensions are not supported")

    (_, _, torch_input) = get_tensors(input_shape, device)
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(input_shape, device)
    if not use_provide_output:
        tt_input_grad = None

    torch_output = torch.cumsum(torch_input, dim)
    torch_output.backward(torch_output_grad)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_input_grad_cpu = (
        ttnn.experimental.operations.primary.moreh_cumsum_backward(
            tt_output_grad, dim=dim, input_grad=tt_input_grad, compute_kernel_config=compute_kernel_config
        )
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
