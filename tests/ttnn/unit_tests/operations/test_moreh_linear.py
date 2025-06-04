# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_ttnn,
)


def get_tensors(
    input_shape,
    mat2_shape,
    output_shape,
    require_input_grad,
    require_mat2_grad,
    device,
    *,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
    use_randint=True,
    is_1d=False,
    low_int=-2,
    high_int=3,
):
    """
    Returns tensors for input, mat2, output, and their gradients (if required), both in ttnn and torch:
    0. ttnn - tilized input tensor
    1. ttnn - tilized mat2 tensor
    2. ttnn - tilized output tensor
    3. ttnn - tilized output gradient tensor (if required), otherwise None
    4. ttnn - tilized input gradient tensor (if required), otherwise None
    5. ttnn - tilized mat2 gradient tensor (if required), otherwise None
    6. torch input tensor
    7. torch mat2 tensor
    8. torch output gradient tensor (if required), otherwise None
    """
    if use_randint:
        tensors = [
            torch.randint(low=low_int, high=high_int, size=input_shape, dtype=torch_dtype),  # input
            torch.randint(low=low_int, high=high_int, size=mat2_shape, dtype=torch_dtype),  # other
            torch.randint(low=low_int, high=high_int, size=output_shape, dtype=torch_dtype),  # output
            (
                torch.randint(low=low_int, high=high_int, size=output_shape, dtype=torch_dtype)
                if (require_input_grad or require_mat2_grad)
                else None
            ),  # output_grad
            torch.full(input_shape, float("nan"), dtype=torch_dtype) if require_input_grad else None,  # input_grad
            torch.full(mat2_shape, float("nan"), dtype=torch_dtype) if require_mat2_grad else None,  # other_grad
        ]
    else:
        tensors = [
            torch.rand(input_shape, dtype=torch_dtype),  # input
            torch.rand(mat2_shape, dtype=torch_dtype),  # other
            torch.rand(output_shape, dtype=torch_dtype),  # output
            (
                torch.rand(output_shape, dtype=torch_dtype) if (require_input_grad or require_mat2_grad) else None
            ),  # output_grad
            torch.full(input_shape, float("nan"), dtype=torch_dtype) if require_input_grad else None,  # input_grad
            torch.full(mat2_shape, float("nan"), dtype=torch_dtype) if require_mat2_grad else None,  # other_grad
        ]

    if is_1d:
        tensors[0] = tensors[0].reshape(-1)
        tensors[1] = tensors[1].reshape(-1)

    ttnn_tensors = [
        ttnn.from_torch(tensor, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
        if tensor is not None
        else None
        for tensor in tensors
    ]

    return (*ttnn_tensors, tensors[0], tensors[1], tensors[3])


def get_bias_tensors(
    bias_shape,
    require_bias_grad,
    device,
    *,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
    use_randint=True,
    low_int=-10,
    high_int=10,
):
    """
    Returns tensors for bias and bias_grad (if required):
    0. ttnn - tilized bias tensor
    1. torch - bias tensor
    2. ttnn - tilized bias grad tensor
    """
    if use_randint:
        torch_bias = torch.randint(low=low_int, high=high_int, size=bias_shape, dtype=torch_dtype)
    else:
        torch_bias = torch.rand(size=bias_shape, dtype=torch_dtype) * 10 - 5

    tt_bias = ttnn.from_torch(torch_bias, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)

    tt_bias_grad = None
    if require_bias_grad:
        bias_grad = torch.full(bias_shape, float("nan"), dtype=torch_dtype)
        tt_bias_grad = ttnn.from_torch(bias_grad, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
    return tt_bias, torch_bias, tt_bias_grad


def moreh_linear(shapes, has_bias, has_output, compute_kernel_config, device, npu_dtype=ttnn.bfloat16):
    torch.manual_seed(3072)
    cpu_dtype = torch.bfloat16
    input_shape, weight_shape, bias_shape, output_shape = shapes
    tt_input, tt_weight, _, _, _, _, torch_input, torch_weight, _ = get_tensors(
        input_shape,
        weight_shape,
        output_shape,
        False,
        False,
        is_1d=False,
        device=device,
        ttnn_dtype=npu_dtype,
        torch_dtype=cpu_dtype,
    )

    torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    tt_output = ttnn.from_torch(torch_output, npu_dtype, device=device, layout=ttnn.TILE_LAYOUT) if has_output else None

    if has_bias:
        tt_bias, torch_bias, _ = get_bias_tensors(
            bias_shape, False, device, torch_dtype=cpu_dtype, ttnn_dtype=npu_dtype
        )
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
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["BFP8", "BFP16"])
def test_moreh_linear(shapes, has_bias, compute_kernel_options, npu_dtype, device):
    if npu_dtype == ttnn.bfloat8_b:
        # FAILED test cases produce 0.0 and also precision results.
        pytest.skip("Moreh Linear does not support bfloat8_b")
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
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["BFP8", "BFP16"])
def test_moreh_linear_wo_output(shapes, has_bias, npu_dtype, device):
    if npu_dtype == ttnn.bfloat8_b:
        # FAILED test cases produce 0.0 and also precision results.
        pytest.skip("Moreh Linear does not support bfloat8_b")
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
        input_shape,
        weight_shape,
        output_shape,
        requires_input_grad,
        requires_weight_grad,
        is_1d=False,
        device=device,
        ttnn_dtype=npu_dtype,
        torch_dtype=cpu_dtype,
    )

    tt_bias, torch_bias, tt_bias_grad = get_bias_tensors(
        bias_shape, requires_bias_grad, device, torch_dtype=cpu_dtype, ttnn_dtype=npu_dtype
    )

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
    if requires_input_grad:
        ttcpu_input_grad = ttnn.to_torch(tt_input_grad).to(cpu_dtype)
        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_input_grad is None

    if requires_weight_grad:
        ttcpu_weight_grad = ttnn.to_torch(tt_weight_grad).to(cpu_dtype)
        passing, output_pcc = comp_allclose_and_pcc(
            torch_weight.grad, ttcpu_weight_grad, pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"weight_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_weight_grad is None

    if requires_bias_grad:
        ttcpu_bias_grad = ttnn.to_torch(tt_bias_grad).to(cpu_dtype)
        passing, output_pcc = comp_allclose_and_pcc(torch_bias.grad, ttcpu_bias_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"bias_grad passing={passing} pcc={output_pcc}")
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
@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["BFP8", "BFP16"])
def test_moreh_linear_backward(shapes, requires_grads, requires_bias_grad, compute_kernel_options, npu_dtype, device):
    if npu_dtype == ttnn.bfloat8_b:
        pytest.skip("Moreh Linear Backward does not support bfloat8_b")
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
    cpu_dtype = torch.bfloat16

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
        is_1d=False,
        device=device,
        ttnn_dtype=npu_dtype,
        torch_dtype=cpu_dtype,
        use_randint=False,
    )
    tt_bias, torch_bias, tt_bias_grad = get_bias_tensors(
        bias_shape, requires_bias_grad, device, torch_dtype=cpu_dtype, ttnn_dtype=npu_dtype, use_randint=False
    )
    (_, _, _, _, tt_input_grad_fp32, _, _, _, _) = get_tensors(
        input_shape,
        weight_shape,
        output_shape,
        requires_input_grad,
        requires_weight_grad,
        is_1d=False,
        device=device,
        ttnn_dtype=npu_dtype,
        torch_dtype=cpu_dtype,
        use_randint=False,
    )
    (_, _, tt_bias_grad_fp32) = get_bias_tensors(
        bias_shape, requires_bias_grad, device, torch_dtype=cpu_dtype, ttnn_dtype=npu_dtype, use_randint=False
    )
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
