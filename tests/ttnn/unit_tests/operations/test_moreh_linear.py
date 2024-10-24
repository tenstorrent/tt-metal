# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import comp_allclose_and_pcc, skip_for_grayskull
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_ttnn,
)


def create_tt_tensor(tensor: torch.Tensor, device, npu_dtype):
    return ttnn.from_torch(tensor, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)


def get_tensors(
    input_shape,
    other_shape,
    output_shape,
    require_input_grad,
    require_other_grad,
    is_1d,
    device,
    npu_dtype=ttnn.bfloat16,
    use_randint=True,
):
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

    tt_input = create_tt_tensor(input, device, npu_dtype)
    tt_other = create_tt_tensor(other, device, npu_dtype)
    tt_output = create_tt_tensor(output, device, npu_dtype)

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


def get_bias_tensors(bias_shape, require_bias_grad, device, npu_dtype=ttnn.bfloat16, use_int=True):
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


def moreh_linear(shapes, has_bias, has_output, compute_kernel_config, device, npu_dtype=ttnn.bfloat16):
    torch.manual_seed(3072)
    input_shape, weight_shape, bias_shape, output_shape = shapes
    _, _, _, _, _, _, torch_input, torch_weight, _ = get_tensors(
        input_shape, weight_shape, output_shape, False, False, False, device, npu_dtype
    )
    tt_input = ttnn.from_torch(torch_input, npu_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, npu_dtype, device=device, layout=ttnn.TILE_LAYOUT)

    npu_layout = ttnn.TILE_LAYOUT
    cpu_dtype = torch.bfloat16
    torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(1).to(npu_layout).to(device) if has_output else None

    if has_bias:
        torch_bias = torch.randint(-10, 10, bias_shape, dtype=cpu_dtype)
        tt_bias = ttnn.from_torch(torch_bias, npu_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        tt_bias, torch_bias = None, None
    ## TT Op
    tt_output = ttnn.operations.moreh.linear(
        tt_input, tt_weight, bias=tt_bias, output=tt_output, compute_kernel_config=compute_kernel_config
    )

    ## reference
    torch_output = torch.nn.functional.linear(torch_input, torch_weight, torch_bias)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    ttcpu_output = ttnn.to_torch(tt_output).to(cpu_dtype)
    passing, output_pcc = comp_allclose_and_pcc(torch_output, ttcpu_output, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Passing = {passing}")
    logger.debug(f"Output PCC = {output_pcc}")
    return passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([31, 31], [30, 31], [1, 30], [31, 30]),
        ([31, 31], [30, 31], [1, 1], [31, 30]),
        ([4, 4, 2, 31], [30, 31], [1, 30], [4, 4, 2, 30]),
        ([4, 4, 2, 31], [30, 31], [1, 1], [4, 4, 2, 30]),
        ([2, 2047], [1023, 2047], [1, 1023], [2, 1023]),
        ([2, 2047], [1023, 2047], [1, 1], [2, 1023]),
        ([32, 64], [1024, 64], [1, 1024], [32, 1024]),
        ([32, 64], [1024, 64], [1, 1], [32, 1024]),
        ([3, 32, 1023], [2047, 1023], [1, 2047], [3, 32, 2047]),
        ([3, 32, 1023], [2047, 1023], [1, 1], [3, 32, 2047]),
        ([2, 4, 4, 1024], [2047, 1024], [1, 2047], [2, 4, 4, 2047]),
        ([2, 4, 4, 1024], [2047, 1024], [1, 1], [2, 4, 4, 2047]),
        ([2, 1, 2, 3, 2, 2, 96, 95], [511, 95], [1, 1], [2, 1, 2, 3, 2, 2, 96, 511]),
        ([2, 1, 2, 3, 2, 2, 96, 95], [511, 95], [1, 511], [2, 1, 2, 3, 2, 2, 96, 511]),
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat16], ids=["BFP16"])
def test_moreh_linear(shapes, has_bias, compute_kernel_options, npu_dtype, device):
    torch.manual_seed(3072)
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    passing = moreh_linear(shapes, has_bias, True, compute_kernel_config, device, npu_dtype)
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([32, 32], [32, 32], [1, 1], [32, 32]),
        ([32, 32], [32, 32], [1, 32], [32, 32]),
        ([32, 64], [1024, 64], [1, 1024], [32, 1024]),
        ([32, 64], [1024, 64], [1, 1], [32, 1024]),
        ([2, 1, 2, 3, 2, 2, 96, 96], [512, 96], [1, 1], [2, 1, 2, 3, 2, 2, 96, 512]),
        ([2, 1, 2, 3, 2, 2, 96, 96], [512, 96], [1, 512], [2, 1, 2, 3, 2, 2, 96, 512]),
        # These tests below doesn't pass. They produce incorrect results, 0.0 and NaN.
        # ([4, 4, 4, 32], [32, 32], [1, 32], [4, 4, 4, 32]),
        # ([4, 4, 4, 32], [32, 32], [1, 1], [4, 4, 4, 32]),
        # ([2, 2047], [1024, 2047], [1, 1024], [2, 1024]),
        # ([2, 2047], [1024, 2047], [1, 1], [2, 1024]),
        # ([3, 32, 1024], [2048, 1024], [1, 2048], [3, 32, 2048]),
        # ([3, 32, 1024], [2048, 1024], [1, 1], [3, 32, 2048]),
        # ([2, 4, 4, 1024], [2048, 1024], [1, 2048], [2, 4, 4, 2048]),
        # ([2, 4, 4, 1024], [2048, 1024], [1, 1], [2, 4, 4, 2048]),
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat8_b], ids=["BFP8"])
def test_moreh_linear_bfp8(shapes, has_bias, compute_kernel_options, npu_dtype, device):
    torch.manual_seed(3072)
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    passing = moreh_linear(shapes, has_bias, True, compute_kernel_config, device, npu_dtype)
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([31, 31], [30, 31], [1, 30], [31, 30]),
        ([2, 1, 2, 3, 2, 2, 96, 95], [511, 95], [1, 1], [2, 1, 2, 3, 2, 2, 96, 511]),
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat16], ids=["BFP16"])
def test_moreh_linear_wo_output(shapes, has_bias, npu_dtype, device):
    torch.manual_seed(3072)
    compute_kernel_config = get_compute_kernel_options(False)
    passing = moreh_linear(shapes, has_bias, False, compute_kernel_config, device, npu_dtype)
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([32, 32], [32, 32], [1, 32], [32, 32]),
        ([2, 1, 2, 3, 2, 2, 96, 96], [512, 96], [1, 1], [2, 1, 2, 3, 2, 2, 96, 512]),
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat8_b], ids=["BFP8"])
def test_moreh_linear_wo_output_bfp8(shapes, has_bias, npu_dtype, device):
    torch.manual_seed(3072)
    compute_kernel_config = get_compute_kernel_options(False)
    passing = moreh_linear(shapes, has_bias, False, compute_kernel_config, device, npu_dtype)
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([31, 31], [30, 31], [1, 1], [31, 30]),
        ([2, 1, 2, 3, 2, 2, 96, 95], [511, 95], [1, 511], [2, 1, 2, 3, 2, 2, 96, 511]),
    ),
)
def test_moreh_linear_enable_cache(shapes, device, use_program_cache):
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        passing = moreh_linear(shapes, True, True, get_compute_kernel_options(False), device)
        assert passing
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


def moreh_linear_backward(
    shapes,
    requires_input_grad,
    requires_weight_grad,
    requires_bias_grad,
    compute_kernel_config,
    device,
    npu_dtype=ttnn.bfloat16,
):
    input_shape, weight_shape, bias_shape, output_shape = shapes
    if not requires_input_grad and not requires_weight_grad and not requires_bias_grad:
        pytest.skip("At least one grad is requires")

    cpu_dtype = torch.bfloat16

    (
        tt_input,
        tt_weight,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_weight_grad,
        torch_input,
        torch_weight,
        torch_output_grad,
    ) = get_tensors(
        input_shape, weight_shape, output_shape, requires_input_grad, requires_weight_grad, False, device, npu_dtype
    )

    # tt_bias, torch_bias, tt_bias_grad = get_bias_tensors(bias_shape, requires_bias_grad, device, npu_dtype)
    torch_bias = torch.randint(-10, 10, bias_shape, dtype=cpu_dtype)
    tt_bias = ttnn.from_torch(torch_bias, npu_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias_grad = None
    if requires_bias_grad:
        bias_grad = torch.full(bias_shape, float("nan"), dtype=cpu_dtype)
        tt_bias_grad = ttnn.from_torch(bias_grad, npu_dtype, device=device, layout=ttnn.TILE_LAYOUT)

    ## tt linear backward
    tt_input_grad, tt_weight_grad, tt_bias_grad = ttnn.operations.moreh.linear_backward(
        tt_output_grad,
        tt_input,
        tt_weight,
        are_required_outputs=(requires_input_grad, requires_weight_grad, requires_bias_grad),
        bias=tt_bias,
        input_grad=tt_input_grad,
        weight_grad=tt_weight_grad,
        bias_grad=tt_bias_grad,
        compute_kernel_config=compute_kernel_config,
    )
    ## reference
    torch_output = torch.nn.functional.linear(
        torch_input.requires_grad_(requires_input_grad),
        torch_weight.requires_grad_(requires_weight_grad),
        torch_bias.requires_grad_(requires_bias_grad),
    )
    torch_output.backward(torch_output_grad)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    if requires_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_input_grad is None

    if requires_weight_grad:
        ttcpu_weight_grad = tt_weight_grad.cpu().to(cpu_layout).unpad_from_tile(weight_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(
            torch_weight.grad, ttcpu_weight_grad, pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"weight_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_weight_grad is None

    if requires_bias_grad:
        ttcpu_bias_grad = ttnn.to_torch(tt_bias_grad).to(cpu_dtype)
        # ttcpu_bias_grad = tt_bias_grad.cpu().to(cpu_layout).unpad_from_tile(bias_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_bias.grad, ttcpu_bias_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"bias_grad passing={passing} pcc={output_pcc}")
        if not passing:
            print("torch_bias.grad", torch_bias.grad)
            print("ttcpu_bias_grad", ttcpu_bias_grad)
        assert passing
    else:
        assert tt_bias_grad is None
    return passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([31, 31], [30, 31], [1, 30], [31, 30]),
        ([31, 31], [30, 31], [1, 1], [31, 30]),
        ([4, 4, 2, 31], [30, 31], [1, 30], [4, 4, 2, 30]),
        ([4, 4, 2, 31], [30, 31], [1, 1], [4, 4, 2, 30]),
        ([2, 2047], [1023, 2047], [1, 1023], [2, 1023]),
        ([2, 2047], [1023, 2047], [1, 1], [2, 1023]),
        ([32, 64], [1024, 64], [1, 1024], [32, 1024]),
        ([32, 64], [1024, 64], [1, 1], [32, 1024]),
        ([3, 32, 1023], [1536, 1023], [1, 1536], [3, 32, 1536]),
        ([3, 32, 1023], [1536, 1023], [1, 1], [3, 32, 1536]),
        ([2, 4, 4, 1024], [1536, 1024], [1, 1536], [2, 4, 4, 1536]),
        ([2, 4, 4, 1024], [1200, 1024], [1, 1], [2, 4, 4, 1200]),
        ([2, 1, 2, 1, 2, 2, 96, 95], [127, 95], [1, 1], [2, 1, 2, 1, 2, 2, 96, 127]),
        ([2, 1, 2, 3, 2, 2, 96, 95], [127, 95], [1, 127], [2, 1, 2, 3, 2, 2, 96, 127]),
    ),
)
@pytest.mark.parametrize(
    "requires_grads",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
@pytest.mark.parametrize("requires_bias_grad", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat16], ids=["BFP16"])
def test_moreh_linear_backward(shapes, requires_grads, requires_bias_grad, compute_kernel_options, npu_dtype, device):
    torch.manual_seed(3072)
    requires_input_grad, requires_weight_grad = requires_grads
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    passing = moreh_linear_backward(
        shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, compute_kernel_config, device, npu_dtype
    )
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        # Only pass with require_bias_grad=F
        ([32, 32], [32, 32], [1, 32], [32, 32]),
        ([32, 32], [32, 32], [1, 1], [32, 32]),
        # Only pass with require_bias_grad=F and requires_grad=(T, F)
        # ([2, 1, 2, 1, 2, 2, 96, 96], [128, 96], [1, 1], [2, 1, 2, 1, 2, 2, 96, 128]),
        # ([2, 1, 2, 3, 2, 2, 96, 96], [128, 96], [1, 128], [2, 1, 2, 3, 2, 2, 96, 128]),
        # Only pass with require_bias_grad=F and requires_grad=(F, T)
        # ([32, 64], [1024, 64], [1, 1024], [32, 1024]),
        # ([32, 64], [1024, 64], [1, 1], [32, 1024]),
        # ([3, 32, 1024], [1536, 1024], [1, 1536], [3, 32, 1536]),
        # ([3, 32, 1024], [1536, 1024], [1, 1], [3, 32, 1536]),
        # Fail all tests
        # ([4, 4, 2, 32], [32, 32], [1, 32], [4, 4, 2, 32]),
        # ([4, 4, 2, 32], [32, 32], [1, 1], [4, 4, 2, 32]),
        # ([2, 2048], [1024, 2048], [1, 1024], [2, 1024]),
        # ([2, 2048], [1024, 2048], [1, 1], [2, 1024]),
        # ([2, 4, 4, 1024], [1536, 1024], [1, 1536], [2, 4, 4, 1536]),
        # ([2, 4, 4, 1024], [1216, 1024], [1, 1], [2, 4, 4, 1216]),
    ),
)
@pytest.mark.parametrize(
    "requires_grads",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
@pytest.mark.parametrize("requires_bias_grad", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat8_b], ids=["BFP8"])
def test_moreh_linear_backward_bfp8(
    shapes, requires_grads, requires_bias_grad, compute_kernel_options, npu_dtype, device
):
    torch.manual_seed(3072)
    if requires_bias_grad:
        # Right now bfp8 will fail if requires_bias_grad==True because moreh_bias_add doesn't support bfp8
        pytest.skip()
    requires_input_grad, requires_weight_grad = requires_grads
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    passing = moreh_linear_backward(
        shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, compute_kernel_config, device, npu_dtype
    )
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([31, 31], [30, 31], [1, 30], [31, 30]),
        ([31, 31], [30, 31], [1, 1], [31, 30]),
        ([2, 4, 4, 1024], [1536, 1024], [1, 1536], [2, 4, 4, 1536]),
        ([32, 1023], [1536, 1023], [1, 1], [32, 1536]),
    ),
)
def test_moreh_linear_backward_enable_cache(shapes, device, use_program_cache):
    requires_input_grad, requires_weight_grad, requires_bias_grad = (True, True, True)
    compute_kernel_config = get_compute_kernel_options(False)

    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        passing = moreh_linear_backward(
            shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, compute_kernel_config, device
        )
        assert passing
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@skip_for_grayskull("GS does not support fp32")
@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        # GPT2-Small cases
        ([8, 512, 768], [2304, 768], [1, 2304], [8, 512, 2304]),
        ([8, 512, 768], [768, 768], [1, 768], [8, 512, 768]),
        ([8, 512, 768], [3072, 768], [1, 3072], [8, 512, 3072]),
    ),
)
def test_moreh_bias_backward_fp32(shapes, device):
    torch.manual_seed(3072)
    compute_kernel_fp32_config = get_compute_kernel_options(True)
    compute_kernel_config = get_compute_kernel_options(False)
    requires_input_grad, requires_weight_grad, requires_bias_grad = (True, False, True)
    npu_dtype = ttnn.bfloat16
    input_shape, weight_shape, bias_shape, output_shape = shapes
    (
        tt_input,
        tt_weight,
        _,
        tt_output_grad,
        tt_input_grad,
        _,
        torch_input,
        torch_weight,
        torch_output_grad,
    ) = get_tensors(
        input_shape,
        weight_shape,
        output_shape,
        requires_input_grad,
        requires_weight_grad,
        False,
        device,
        npu_dtype,
        False,
    )
    tt_bias, torch_bias, tt_bias_grad = get_bias_tensors(bias_shape, requires_bias_grad, device, npu_dtype, False)
    (_, _, _, _, tt_input_grad_fp32, _, _, _, _) = get_tensors(
        input_shape,
        weight_shape,
        output_shape,
        requires_input_grad,
        requires_weight_grad,
        False,
        device,
        npu_dtype,
        False,
    )
    (_, _, tt_bias_grad_fp32) = get_bias_tensors(bias_shape, requires_bias_grad, device, npu_dtype, False)
    ## tt linear backward (fp32 mode)
    tt_input_grad_fp32, _, tt_bias_grad_fp32 = ttnn.operations.moreh.linear_backward(
        tt_output_grad,
        tt_input,
        tt_weight,
        are_required_outputs=(requires_input_grad, requires_weight_grad, requires_bias_grad),
        bias=tt_bias,
        input_grad=tt_input_grad_fp32,
        weight_grad=None,
        bias_grad=tt_bias_grad_fp32,
        compute_kernel_config=compute_kernel_fp32_config,
    )
    ## tt linear backward (bf16 mode)
    tt_input_grad, _, tt_bias_grad = ttnn.operations.moreh.linear_backward(
        tt_output_grad,
        tt_input,
        tt_weight,
        are_required_outputs=(requires_input_grad, requires_weight_grad, requires_bias_grad),
        bias=tt_bias,
        input_grad=tt_input_grad,
        weight_grad=None,
        bias_grad=tt_bias_grad,
        compute_kernel_config=compute_kernel_config,
    )
    torch_input_fp32 = torch_input.float()
    torch_weight_fp32 = torch_weight.float()
    torch_bias_fp32 = torch_bias.float()
    ## reference
    torch_output = torch.nn.functional.linear(
        torch_input_fp32.requires_grad_(requires_input_grad),
        torch_weight_fp32.requires_grad_(requires_weight_grad),
        torch_bias_fp32.requires_grad_(requires_bias_grad),
    )
    torch_output.backward(torch_output_grad.float())
    ## test for equivalance
    rtol = atol = 0.1
    tt_bias_grad_fp32_cpu = tt_bias_grad_fp32.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(bias_shape).to_torch()
    tt_bias_grad_cpu = tt_bias_grad.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(bias_shape).to_torch()
    passing, output_pcc = comp_allclose_and_pcc(
        torch_bias_fp32.grad, tt_bias_grad_fp32_cpu, pcc=0.98, rtol=rtol, atol=atol
    )
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing
    diff_fp32 = torch.abs(torch_bias_fp32.grad - tt_bias_grad_fp32_cpu)
    logger.debug(f"std={torch.std(diff_fp32)}")
    logger.debug(f"mean={diff_fp32.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff_fp32.reshape(-1), 5)}")
    diff = torch.abs(torch_bias_fp32.grad - tt_bias_grad_cpu)
    logger.debug(f"std={torch.std(diff)}")
    logger.debug(f"mean={diff.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff.reshape(-1), 5)}")
    assert diff_fp32.mean() < diff.mean()
