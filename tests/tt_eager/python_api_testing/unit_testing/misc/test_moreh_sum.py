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


def get_backward_tensors(output_grad_shape, input_grad_shape, device, *, with_padding=True, use_randint=True):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    if use_randint:
        torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_grad = torch.randint(-2, 3, input_grad_shape, dtype=cpu_dtype)
    else:
        torch_output_grad = torch.rand(output_grad_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_grad = torch.rand(input_grad_shape, dtype=cpu_dtype)

    if with_padding:
        tt_output_grad = (
            ttl.tensor.Tensor(torch_output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        )
        tt_input_grad = (
            ttl.tensor.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        )
    else:
        tt_output_grad = ttl.tensor.Tensor(torch_output_grad, npu_dtype).to(npu_layout).to(device)
        tt_input_grad = ttl.tensor.Tensor(torch_input_grad, npu_dtype).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
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
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("use_provide_output", (True, False), ids=["True", "False"])
def test_moreh_sum(input_shape, dims, compute_kernel_options, use_provide_output, device):
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
        ttl.operations.primary.moreh_sum(
            tt_input, dims=dims, output=tt_output, compute_kernel_config=compute_kernel_config
        )
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


def reduce_rows(x, dims):
    index_tuple = tuple(slice(0, 1) if i in dims else slice(None) for i in range(x.dim()))
    return x[index_tuple]


@pytest.mark.parametrize(
    "input_shape",
    (
        ([TILE_HEIGHT, TILE_WIDTH]),
        ([TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([2, 3, 2, 4, TILE_HEIGHT * 4, TILE_WIDTH * 4]),
        ([3, 2, 4, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "TILE_HEIGHT, TILE_WIDTH",
        "TILE_HEIGHT - 1, TILE_WIDTH - 1",
        "2, 3, 2, 4, TILE_HEIGHT * 4, TILE_WIDTH * 4",
        "3, 2, 4, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
    ),
    ids=["0", "1", "2", "3", "4", "5"],
)
@pytest.mark.parametrize("use_provide_output", (True, False), ids=["True", "False"])
def test_moreh_sum_non_4d(input_shape, dims, use_provide_output, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    input_rank = len(input_shape)
    for dim in dims:
        if dim >= input_rank:
            pytest.skip(f"input dim {dim} exceeds the dims of input tensor {len(input_shape)}.")

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device)
    if not use_provide_output:
        tt_output = None

    compute_kernel_config = get_compute_kernel_options(False)

    torch_output = torch.sum(torch_input, dims, True)
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.moreh_sum(
            tt_input, dims=dims, output=tt_output, compute_kernel_config=compute_kernel_config
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )
    rtol = atol = 0.12
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "input_shape",
    (
        [10, TILE_HEIGHT * 12, TILE_WIDTH * 12],
        [10, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1],
    ),
    ids=[
        "10, TILE_HEIGHT * 12, TILE_WIDTH * 12",
        "10, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    ([0], [1], [2]),
    ids=["dim-n", "dim-h", "dim-w"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_fp32_dest_acc(input_shape, dims, compute_kernel_options, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device, use_randint=False)
    torch_input = torch_input.float()
    torch_output = torch.sum(torch_input, dims, True)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.moreh_sum(
            tt_input, dims=dims, output=tt_output, compute_kernel_config=compute_kernel_config
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    logger.debug(f"std={torch.std(torch.abs(torch_output - tt_output_cpu))}")
    logger.debug(f"mean={torch.abs(torch_output - tt_output_cpu).mean()}")

    # TODO
    # assert passing


@pytest.mark.parametrize(
    "input_shape",
    (([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1]),),
    ids=[
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1",
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
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("use_provide_input_grad", (True, False), ids=["True", "False"])
def test_moreh_sum_backward(input_shape, dims, compute_kernel_options, use_provide_input_grad, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, _, torch_input) = get_tensors(input_shape, output_shape, device)
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(output_shape, input_shape, device)

    if not use_provide_input_grad:
        tt_input_grad = None

    torch_output = torch.sum(torch_input, dims, True)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_sum_backward(
            tt_output_grad, tt_input, dims=dims, input_grad=tt_input_grad, compute_kernel_config=compute_kernel_config
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


@pytest.mark.parametrize(
    "input_shape",
    ([2, 3, 2, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1],),
    ids=[
        "2, 3, 2, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    ([0], [4], [5], [4, 5], [1, 4, 5]),
    ids=["dim-n", "dim-h", "dim-w", "dim-hw", "dim-nhw"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_backward_fp32_dest_acc(input_shape, dims, compute_kernel_options, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, _, torch_input) = get_tensors(input_shape, output_shape, device, use_randint=False)
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(
        output_shape, input_shape, device, use_randint=False
    )

    # convert torch_input to float32 dtype
    torch_input = torch_input.detach().clone().to(dtype=torch.float32).requires_grad_(True)
    torch_output_grad = torch_output_grad.float()
    torch_output = torch.sum(torch_input, dims, True)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_sum_backward(
            tt_output_grad, tt_input, dims=dims, input_grad=tt_input_grad, compute_kernel_config=compute_kernel_config
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(input_shape)
        .to_torch()
    )

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    logger.debug(f"std={torch.std(torch.abs(torch_input.grad- tt_input_grad_cpu))}")
    logger.debug(f"mean={torch.abs(torch_input.grad - tt_input_grad_cpu).mean()}")
    assert passing
