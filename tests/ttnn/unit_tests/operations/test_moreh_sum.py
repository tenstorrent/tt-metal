# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger


import ttnn
from models.utility_functions import (
    comp_allclose_and_pcc,
    skip_for_grayskull,
)
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    filter_indices,
    filter_indices_with_last_two,
    TILE_HEIGHT,
    TILE_WIDTH,
)


def is_npu_dtype_uint32(data_type):
    return data_type == ttnn.uint32


def is_npu_dtype_float(data_type):
    return data_type == ttnn.float32 or data_type == ttnn.bfloat16


def get_tensors(
    input_shape,
    dim,
    device,
    *,
    with_padding=True,
    use_randint=True,
    keepdim=False,
    npu_dtype=ttnn.bfloat16,
    cpu_dtype=torch.bfloat16,
):
    npu_layout = ttnn.TILE_LAYOUT
    output_shape = input_shape.copy()
    if dim is None or dim == []:
        dim = list(range(len(input_shape)))

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        output_shape[d] = 1

    if keepdim:
        torch_output_shape = output_shape.copy()
        tt_output_shape = output_shape.copy()
    else:
        torch_output_shape = filter_indices(output_shape, dim)
        tt_output_shape = filter_indices_with_last_two(output_shape, dim)

    if use_randint:
        int_min = 0 if is_npu_dtype_uint32(npu_dtype) else -2
        int_max = 10 if is_npu_dtype_uint32(npu_dtype) else 3
        requires_grad = True if is_npu_dtype_float(npu_dtype) else False
        torch_input = torch.randint(int_min, int_max, input_shape, dtype=cpu_dtype, requires_grad=requires_grad)
        torch_output = torch.randint(int_min, int_max, tt_output_shape, dtype=cpu_dtype)
    else:
        torch_input = torch.rand(input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.rand(tt_output_shape, dtype=cpu_dtype)

    if with_padding:
        tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        tt_input = ttnn.Tensor(torch_input, npu_dtype).to(npu_layout).to(device)
        tt_output = ttnn.Tensor(torch_output, npu_dtype).to(npu_layout).to(device)

    return tt_input, tt_output, tt_output_shape, torch_output_shape, torch_input


def get_backward_tensors(
    tt_output_grad_shape,
    torch_output_grad_shape,
    torch_input_grad_shape,
    device,
    *,
    with_padding=True,
    use_randint=True,
):
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    if use_randint:
        torch_output_grad = torch.randint(-2, 3, torch_output_grad_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_grad = torch.randint(-2, 3, torch_input_grad_shape, dtype=cpu_dtype)
    else:
        torch_output_grad = torch.rand(torch_output_grad_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_grad = torch.rand(torch_input_grad_shape, dtype=cpu_dtype)

    if with_padding:
        tt_output_grad = (
            ttnn.Tensor(torch_output_grad.reshape(tt_output_grad_shape), npu_dtype)
            .pad_to_tile(float("nan"))
            .to(npu_layout)
            .to(device)
        )
        tt_input_grad = ttnn.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        tt_output_grad = (
            ttnn.Tensor(torch_output_grad.reshape(tt_output_grad_shape), npu_dtype).to(npu_layout).to(device)
        )
        tt_input_grad = ttnn.Tensor(torch_input_grad, npu_dtype).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


def moreh_sum(input_shape, dim, keepdim, use_provide_output, compute_kernel_options, device):
    (tt_input, tt_output, output_shape, _, torch_input) = get_tensors(input_shape, dim, device, keepdim=keepdim)
    torch_output = torch.sum(torch_input, dim, keepdim)

    if not use_provide_output:
        tt_output = None

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = (
        ttnn.operations.moreh.sum(
            tt_input,
            dim,
            keepdim=keepdim,
            output=tt_output,
            compute_kernel_config=compute_kernel_config,
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # test for equivalance
    # TODO(Dongjin) : check while changing rtol after enabling fp32_dest_acc_en
    rtol = atol = 0.12
    passing, output_pcc = comp_allclose_and_pcc(
        torch_output if keepdim else torch_output.reshape(-1),
        tt_output_cpu if keepdim else tt_output_cpu.reshape(-1),
        pcc=0.999,
        rtol=rtol,
        atol=atol,
    )

    logger.debug(f"input_shape={input_shape}, dim={dim}, tt_output_shape={tt_output_cpu.shape}")
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    return passing


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (
        None,
        0,
        1,
        2,
        3,
        [],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2],
        [1, 2, 3],
        [1, 3],
        [2, 3],
    ),
    ids=["None", "0", "1", "2", "3", "[]", "0,1", "0,1,2", "0,1,2,3", "0,1,3", "0,2,3", "1,2", "1,2,3", "1,3", "2,3"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("keepdim", (True, False), ids=["keepdim-true", "keepdim-false"])
def test_moreh_sum(input_shape, dim, keepdim, compute_kernel_options, device):
    torch.manual_seed(2023)
    passing = moreh_sum(input_shape, dim, keepdim, True, compute_kernel_options, device)
    assert passing


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
    "dim",
    (0, 1, 2, 3, 4, 5),
    ids=["0", "1", "2", "3", "4", "5"],
)
@pytest.mark.parametrize("use_provide_output", (True, False), ids=["True", "False"])
@pytest.mark.parametrize("keepdim", (True, False), ids=["keepdim-true", "keepdim-false"])
def test_moreh_sum_non_4d(input_shape, dim, keepdim, use_provide_output, device):
    torch.manual_seed(2023)
    input_rank = len(input_shape)
    if dim >= input_rank:
        pytest.skip(f"input dim {dim} exceeds the dims of input tensor {len(input_shape)}.")

    passing = moreh_sum(input_shape, dim, keepdim, use_provide_output, False, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    [
        [4, TILE_HEIGHT * 4, TILE_WIDTH * 4],
    ],
    ids=[
        "4, TILE_HEIGHT * 4, TILE_WIDTH * 4",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2),
    ids=["0", "1", "2"],
)
def test_moreh_sum_enable_cache(input_shape, dim, device, use_program_cache):
    torch.manual_seed(3072)
    keepdim = [True, False]
    use_provide_output = [True, False]
    for i in range(2):
        passing = moreh_sum(input_shape, dim, keepdim[i], use_provide_output[i], False, device)
        assert passing
    assert device.num_program_cache_entries() == 2


@pytest.mark.parametrize(
    "input_shape",
    (
        [10, TILE_HEIGHT * 12, TILE_WIDTH * 12],
        [10, TILE_HEIGHT * 12 - 10, TILE_WIDTH * 12 - 10],
    ),
    ids=[
        "10, TILE_HEIGHT * 12, TILE_WIDTH * 12",
        "10, TILE_HEIGHT * 12 - 10, TILE_WIDTH * 12 - 10",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2),
    ids=["dim-n", "dim-h", "dim-w"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_fp32_dest_acc(input_shape, dim, compute_kernel_options, device):
    torch.manual_seed(3072)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, tt_output, output_shape, torch_output_shape, torch_input) = get_tensors(
        input_shape, dim, device, use_randint=False, keepdim=True
    )
    torch_input = torch_input.float()
    torch_output = torch.sum(torch_input, dim, True)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = (
        ttnn.operations.moreh.sum(
            tt_input, dim, keepdim=True, output=tt_output, compute_kernel_config=compute_kernel_config
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

    # TODO: Need to check the accuracy for fp32 mode
    # assert passing


def moreh_sum_backward(input_shape, dim, keepdim, use_provide_output, compute_kernel_options, device):
    torch.manual_seed(2023)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, _, tt_output_shape, torch_output_shape, torch_input) = get_tensors(
        input_shape, dim, device, keepdim=keepdim
    )
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(
        tt_output_shape, torch_output_shape, input_shape, device
    )

    if not use_provide_output:
        tt_input_grad = None

    torch_output = torch.sum(torch_input, dim, keepdim)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_input_grad_cpu = (
        ttnn.operations.moreh.sum_backward(
            tt_output_grad,
            input=tt_input,
            dim=dim,
            keepdim=keepdim,
            input_grad=tt_input_grad,
            compute_kernel_config=compute_kernel_config,
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

    return passing


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (
        None,
        0,
        1,
        2,
        3,
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2],
        [1, 2, 3],
        [1, 3],
        [2, 3],
    ),
    ids=["None", "0", "1", "2", "3", "0,1", "0,1,2", "0,1,2,3", "0,1,3", "0,2,3", "1,2", "1,2,3", "1,3", "2,3"],
)
@pytest.mark.parametrize("keepdim", (True, False), ids=["keepdim-true", "keepdim-false"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_backward(input_shape, dim, keepdim, compute_kernel_options, device):
    torch.manual_seed(2023)
    passing = moreh_sum_backward(input_shape, dim, keepdim, True, compute_kernel_options, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (
        None,
        1,
        3,
        [0, 1],
        [0, 2, 3],
        [1, 3],
    ),
    ids=["None", "1", "3", "0,1", "0,2,3", "1,3"],
)
def test_moreh_sum_backward_wo_input_grad(input_shape, dim, device):
    torch.manual_seed(2023)
    passing = moreh_sum_backward(input_shape, dim, True, False, False, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    [
        [4, TILE_HEIGHT * 4, TILE_WIDTH * 4],
    ],
    ids=[
        "4, TILE_HEIGHT * 4, TILE_WIDTH * 4",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2),
    ids=["0", "1", "2"],
)
def test_moreh_sum_backward_enable_cache(input_shape, dim, device, use_program_cache):
    torch.manual_seed(3072)
    keepdim = [True, False]
    use_provide_output = [True, False]
    num_cache_entires = [2, 2, 2]
    for i in range(2):
        passing = moreh_sum_backward(input_shape, dim, keepdim[i], use_provide_output[i], False, device)
        assert passing
    assert device.num_program_cache_entries() == num_cache_entires[dim]


@pytest.mark.parametrize(
    "input_shape",
    ([2, 3, 2, 4, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 6 - 1],),
    ids=[
        "2, 3, 2, 4, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 4, 5, [4, 5], [1, 4, 5]),
    ids=["dim-n", "dim-h", "dim-w", "dim-hw", "dim-nhw"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_backward_fp32_dest_acc(input_shape, dim, compute_kernel_options, device):
    torch.manual_seed(2023)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, _, tt_output_shape, torch_output_shape, torch_input) = get_tensors(
        input_shape, dim, device, use_randint=False
    )
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(
        tt_output_shape, torch_output_shape, input_shape, device, use_randint=False
    )

    # convert torch_input to float32 dtype
    torch_input = torch_input.detach().clone().to(dtype=torch.float32).requires_grad_(True)
    torch_output_grad = torch_output_grad.float()
    torch_output = torch.sum(torch_input, dim)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_input_grad_cpu = (
        ttnn.operations.moreh.sum_backward(
            tt_output_grad,
            input=tt_input,
            dim=dim,
            input_grad=tt_input_grad,
            compute_kernel_config=compute_kernel_config,
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


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shape",
    [
        [TILE_HEIGHT, TILE_WIDTH],
        [3, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1],
        [2, 2, 3, TILE_HEIGHT * 8, TILE_WIDTH * 8],
        [3, TILE_HEIGHT * 20 - 2, TILE_WIDTH * 20 - 2],
        [10, 3, TILE_HEIGHT * 8 - 1, TILE_WIDTH * 8 - 1],
    ],
    ids=[
        "TILE_HEIGHT, TILE_WIDTH",
        "3, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1",
        "2, 2, 3, TILE_HEIGHT * 8, TILE_WIDTH * 8",
        "3, TILE_HEIGHT * 20 - 2, TILE_WIDTH * 20 - 2",
        "10, 3, TILE_HEIGHT * 8 - 1, TILE_WIDTH * 8 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    [-1, -2, 0, 1],
    ids=["dim-w", "dim-h", "dim-b0", "dim-b1"],
)
@pytest.mark.parametrize(
    "data_type",
    [ttnn.int32],
    ids=["int32"],
)
def test_moreh_sum_integer(input_shape, dim, data_type, device):
    if (dim == 0 or dim == 1) and (len(input_shape) - dim <= 2):
        pytest.skip(f"skip sum for batch-dim with this config. {input_shape} and {dim}")

    torch.manual_seed(3072)

    compute_kernel_config = get_compute_kernel_options(True)
    (tt_input, tt_output, tt_output_shape, _, torch_input) = get_tensors(
        input_shape, dim, device, use_randint=True, keepdim=True, npu_dtype=data_type, cpu_dtype=torch.int64
    )

    normalized_dim = dim if dim >= 0 else len(input_shape) + dim

    torch_output = torch.sum(torch_input, normalized_dim, True)
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT

    tt_output = ttnn.operations.moreh.sum(
        tt_input, dim=normalized_dim, keepdim=True, output=tt_output, compute_kernel_config=compute_kernel_config
    )

    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(tt_output_shape).to_torch()
    logger.debug(f"{torch.equal(torch_output, tt_output_cpu)}")

    assert torch.equal(torch_output, tt_output_cpu)
