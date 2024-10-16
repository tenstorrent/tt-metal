# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
import ttnn.operations
from models.utility_functions import comp_allclose_and_pcc, skip_for_grayskull
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


def create_tt_tensor(tensor: torch.Tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def get_tensors(
    input_shape, other_shape, output_shape, require_input_grad, require_other_grad, is_1d, device, use_randint=True
):
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT

    # create tensors for forward
    if use_randint:
        input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
        other = torch.randint(-2, 3, other_shape, dtype=cpu_dtype)
        output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    else:
        input = torch.rand(input_shape, dtype=cpu_dtype)
        other = torch.rand(other_shape, dtype=cpu_dtype)
        output = torch.rand(output_shape, dtype=cpu_dtype)

    tt_input = create_tt_tensor(input, device)
    tt_other = create_tt_tensor(other, device)
    tt_output = create_tt_tensor(output, device)

    torch_input = input.reshape(-1) if is_1d else input
    torch_other = other.reshape(-1) if is_1d else other

    # tensors for backward
    output_grad = tt_output_grad = torch_output_grad = tt_input_grad = tt_other_grad = None
    if require_input_grad or require_other_grad:
        output_grad = (
            torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
            if use_randint
            else torch.rand(output_shape, dtype=cpu_dtype)
        )
        tt_output_grad = ttnn.Tensor(output_grad, npu_dtype).pad_to_tile(float(-1)).to(npu_layout).to(device)
        torch_output_grad = output_grad[0][0][0][0] if is_1d else output_grad

        if require_input_grad:
            input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
            tt_input_grad = ttnn.Tensor(input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

        if require_other_grad:
            other_grad = torch.full(other_shape, float("nan"), dtype=cpu_dtype)
            tt_other_grad = (
                ttnn.Tensor(
                    other_grad,
                    npu_dtype,
                )
                .pad_to_tile(float("nan"))
                .to(npu_layout)
                .to(device)
            )

    return (
        tt_input,
        tt_other,
        tt_output,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    )


def get_bias_tensors(bias_shape, require_bias_grad, device, use_int=True):
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    bias = (
        torch.randint(-10, 10, bias_shape, dtype=cpu_dtype)
        if use_int
        else torch.rand(bias_shape, dtype=cpu_dtype) * 10 - 5
    )
    tt_bias = ttnn.Tensor(bias, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_bias_grad = None
    if require_bias_grad:
        bias_grad = torch.full(bias_shape, float("nan"), dtype=cpu_dtype)
        tt_bias_grad = ttnn.Tensor(bias_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return tt_bias, bias, tt_bias_grad


def moreh_matmul(params, has_output, compute_kernel_config, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, tt_output, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device
    )
    if not has_output:
        tt_output = None

    torch_input = torch_input.transpose(-1, -2) if transpose_input else torch_input
    torch_other = torch_other.transpose(-1, -2) if transpose_other else torch_other

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output = ttnn.operations.moreh.matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        output=tt_output,
        compute_kernel_config=compute_kernel_config,
    )
    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch matmul
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    return passing


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([32, 32], [32, 32], [32, 32], False, False),  # single-core
        ([1024, 128], [128, 1024], [1024, 1024], False, False),  # multi-core
        ([128, 1024], [128, 1024], [1024, 1024], True, False),  # input transpose
        ([1024, 128], [1024, 128], [1024, 1024], False, True),  # other transpose
        ([128, 1024], [1024, 128], [1024, 1024], True, True),  # input, other transpose
        ([1020, 128], [128, 1024], [1020, 1024], False, False),  # input mask
        ([1024, 128], [128, 1020], [1024, 1020], False, False),  # other mask
        ([1020, 310], [310, 1020], [1020, 1020], False, False),  # input, other mask
        ([128, 1020], [128, 1024], [1020, 1024], True, False),  # input mask, transpose
        ([1024, 128], [1020, 128], [1024, 1020], False, True),  # other mask, transpose
        ([310, 1020], [1020, 310], [1020, 1020], True, True),  # input, other mask, transpose
        ([3, 1, 2, 1, 4, 1, 319, 95], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470], False, False),  # batched matmul
        ([2, 319, 95], [2, 1, 3, 4, 1, 95, 470], [2, 1, 3, 4, 2, 319, 470], False, False),  # batched matmul
        ([3, 1, 2, 1, 4, 1, 95, 319], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470], True, False),  # batched matmul
        ([2, 319, 95], [2, 1, 3, 4, 1, 470, 95], [2, 1, 3, 4, 2, 319, 470], False, True),  # batched matmul
        (
            [2, 3, 1, 2, 3, 2, 64, 64],
            [2, 1, 4, 2, 1, 2, 64, 64],
            [2, 3, 4, 2, 3, 2, 64, 64],
            False,
            False,
        ),  # batched matmul
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_matmul(params, compute_kernel_options, device):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    passing = moreh_matmul(params, True, compute_kernel_config, device)
    assert passing


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([32, 32], [32, 32], [32, 32], False, False),  # single-core
        ([3, 1, 2, 1, 4, 1, 95, 319], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470], True, False),  # batched matmul
        ([2, 319, 95], [2, 1, 3, 4, 1, 470, 95], [2, 1, 3, 4, 2, 319, 470], False, True),  # batched matmul
        (
            [2, 3, 1, 2, 3, 2, 64, 64],
            [2, 1, 4, 2, 1, 2, 64, 64],
            [2, 3, 4, 2, 3, 2, 64, 64],
            False,
            False,
        ),  # batched matmul
    ),
)
def test_moreh_matmul_wo_output(params, device):
    passing = moreh_matmul(params, False, None, device)
    assert passing


@pytest.mark.parametrize(
    "params",
    (
        # input, weight, bias(1d or scalar), output
        ([32, 32], [32, 32], [32, 32], False, False),  # single-core
        (
            [2, 3, 1, 2, 3, 2, 64, 64],
            [2, 1, 4, 2, 1, 2, 64, 64],
            [2, 3, 4, 2, 3, 2, 64, 64],
            False,
            False,
        ),  # batched matmul
    ),
)
def test_moreh_matmul_enable_cache(params, device, use_program_cache):
    torch.manual_seed(3072)
    for i in range(4):
        # change input's transpose option
        if i % 2 == 1:
            param_list = list(params)
            param_list[3] = False if param_list[3] else True
            params = tuple(param_list)
        passing = moreh_matmul(params, False, None, device)
        assert passing
    assert device.num_program_cache_entries() == 2


@skip_for_grayskull("GS does not support fp32")
@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([32, 3200], [3200, 32], [32, 32], False, False),
        ([3100, 31], [3100, 31], [31, 31], True, False),
    ),
)
def test_moreh_matmul_fp32_dest_acc(params, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, tt_output_fp32, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device, use_randint=False
    )

    compute_kernel_config_fp32_dest_acc = get_compute_kernel_options(True)
    compute_kernel_config_bf16_dest_acc = get_compute_kernel_options(False)

    torch_input = torch_input.transpose(-1, -2) if transpose_input else torch_input
    torch_other = torch_other.transpose(-1, -2) if transpose_other else torch_other

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_fp32 = ttnn.operations.moreh.matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        output=tt_output_fp32,
        compute_kernel_config=compute_kernel_config_fp32_dest_acc,
    )

    tt_output_fp16 = ttnn.operations.moreh.matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        compute_kernel_config=compute_kernel_config_bf16_dest_acc,
    )

    tt_output_cpu_fp32 = tt_output_fp32.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    tt_output_cpu_bf16 = tt_output_fp16.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch matmul (float)
    torch_out = torch.matmul(torch_input.float(), torch_other.float())

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu_fp32, pcc=0.99, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    diff = torch.abs(torch_out - tt_output_cpu_fp32)
    logger.debug(f"std={torch.std(diff)}")
    logger.debug(f"mean={diff.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff.reshape(-1), 5)}")

    assert passing

    torch_out = torch.matmul(torch_input.bfloat16(), torch_other.bfloat16())
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu_bf16, pcc=0.99, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    diff_fp16 = torch.abs(torch_out - tt_output_cpu_bf16)
    logger.debug(f"std={torch.std(diff_fp16)}")
    logger.debug(f"mean={diff_fp16.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff_fp16.reshape(-1), 5)}")

    assert diff.mean() < diff_fp16.mean()


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
def test_moreh_matmul_1d(input_shape, device):
    torch.manual_seed(3072)
    # get tensors
    output_shape = [1, 1, 1, 1]
    tt_input, tt_other, _, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, input_shape, output_shape, False, False, True, device
    )

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_out = (
        ttnn.operations.moreh.matmul(tt_input, tt_other).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    )

    # torch matmul
    torch_input = torch.reshape(torch_input, (torch_input.shape[-1],))
    torch_other = torch.reshape(torch_other, (torch_other.shape[-1],))
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_out[0][0][0][0], pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape
        ([3, 128, 96], [3, 4, 1, 96, 256], [3, 4, 3, 128, 256]),
        ([3, 3, 313, 511], [3, 3, 511, 765], [3, 3, 313, 765]),
        ([3, 1, 2, 1, 4, 1, 319, 95], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470]),
        ([3, 2, 1, 470, 95], [2, 1, 3, 1, 2, 2, 95, 319], [2, 1, 3, 3, 2, 2, 470, 319]),
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_matmul_backward(params, requires_grad, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape = params
    require_input_grad, require_other_grad = requires_grad

    # get tensors
    (
        tt_input,
        tt_other,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(input_shape, other_shape, output_shape, require_input_grad, require_other_grad, False, device)

    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    tt_input_grad, tt_other_grad = ttnn.operations.moreh.matmul_backward(
        tt_output_grad,
        tt_input,
        tt_other,
        are_required_outputs=(require_input_grad, require_other_grad),
        input_a_grad=tt_input_grad,
        input_b_grad=tt_other_grad,
    )

    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing
    else:
        assert tt_input_grad is None

    if require_other_grad:
        ttcpu_other_grad = tt_other_grad.cpu().to(cpu_layout).unpad_from_tile(other_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_other.grad, ttcpu_other_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing
    else:
        assert tt_other_grad is None


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_matmul_1d_backward(input_shape, requires_grad, device):
    torch.manual_seed(3072)
    require_input_grad, require_other_grad = requires_grad
    output_shape = [1, 1, 1, 1]
    # get tensors
    (
        tt_input,
        tt_other,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(input_shape, input_shape, output_shape, require_input_grad, require_other_grad, True, device)

    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    for _ in range(2):
        ttnn.operations.moreh.matmul_backward(
            tt_output_grad,
            tt_input,
            tt_other,
            are_required_outputs=(require_input_grad, require_other_grad),
            input_a_grad=tt_input_grad,
            input_b_grad=tt_other_grad,
        )

    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(
            torch_input.grad, ttcpu_input_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_other_grad:
        ttcpu_other_grad = tt_other_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(
            torch_other.grad, ttcpu_other_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing


@skip_for_grayskull("GS does not support fp32")
@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([31, 3100], [3100, 31], [31, 31], False, False),
    ),
)
def test_moreh_matmul_with_bias_add_fp32_dest_acc(params, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, tt_output_fp32, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device, use_randint=False
    )
    tt_bias, torch_bias, _ = get_bias_tensors([1, 31], False, device, False)
    compute_kernel_config_fp32_dest_acc = get_compute_kernel_options(True)
    compute_kernel_config_bf16_dest_acc = get_compute_kernel_options(False)
    torch_input = torch_input.transpose(-1, -2) if transpose_input else torch_input
    torch_other = torch_other.transpose(-1, -2) if transpose_other else torch_other
    # tt matmul
    tt_output_fp32 = ttnn.operations.moreh.matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        output=tt_output_fp32,
        bias=tt_bias,
        compute_kernel_config=compute_kernel_config_fp32_dest_acc,
    )
    tt_output_fp16 = ttnn.operations.moreh.matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        bias=tt_bias,
        compute_kernel_config=compute_kernel_config_bf16_dest_acc,
    )
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu_fp32 = tt_output_fp32.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    tt_output_cpu_bf16 = tt_output_fp16.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    # torch matmul (float)
    torch_out = torch.matmul(torch_input.float(), torch_other.float()) + torch_bias
    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu_fp32, pcc=0.99, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    diff = torch.abs(torch_out - tt_output_cpu_fp32)
    logger.debug(f"std={torch.std(diff)}")
    logger.debug(f"mean={diff.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff.reshape(-1), 5)}")
    assert passing
    torch_out = torch.matmul(torch_input.bfloat16(), torch_other.bfloat16())
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu_bf16, pcc=0.99, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    diff_fp16 = torch.abs(torch_out - tt_output_cpu_bf16)
    logger.debug(f"std={torch.std(diff_fp16)}")
    logger.debug(f"mean={diff_fp16.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff_fp16.reshape(-1), 5)}")
    assert diff.mean() < diff_fp16.mean()
