# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc, is_grayskull
from tests.ttnn.unit_tests.operations.test_utils import (
    compute_kernel_ids,
    compute_kernel_options,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
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
    tensors = [
        torch.rand(input_shape, dtype=torch_dtype),
        torch.rand(mat2_shape, dtype=torch_dtype),
        torch.rand(output_shape, dtype=torch_dtype),
        torch.rand(output_shape, dtype=torch_dtype) if (require_input_grad or require_mat2_grad) else None,
        torch.full(input_shape, float("nan"), dtype=torch_dtype) if require_input_grad else None,
        torch.full(mat2_shape, float("nan"), dtype=torch_dtype) if require_mat2_grad else None,
    ]
    ttnn_tensors = [
        create_ttnn_tilized_tensor(tensor, device, ttnn_dtype) if tensor is not None else None for tensor in tensors
    ]
    return (*ttnn_tensors, tensors[0], tensors[1], tensors[3])


def run_moreh_bmm(
    shape,
    optional_output,
    compute_kernel_options,
    device,
    *,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
):
    """
    Function to test Batched Matrix Multiplication (BMM) using PyTorch and TTNN backends.

    Args:
        shape (list): Shape of the input tensors [batch, m, k, n].
        optional_output (bool): Flag indicating if output tensor should be preallocated.
        compute_kernel_options: Configuration options for the compute kernel.
        device: The device on which to run the operations.
    """
    input_shape = [shape[0], shape[1], shape[2]]
    mat2_shape = [shape[0], shape[2], shape[3]]
    output_shape = [shape[0], shape[1], shape[3]]
    (
        ttnn_input,
        ttnn_mat2,
        ttnn_output,
        _,
        _,
        _,
        input,
        mat2,
        _,
    ) = get_tensors(
        input_shape,
        mat2_shape,
        output_shape,
        False,
        False,
        device,
        torch_dtype=torch_dtype,
        ttnn_dtype=ttnn_dtype,
    )
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # Perform PyTorch BMM
    output = torch.bmm(input, mat2)

    # Perform TTNN BMM
    ttnn_output = ttnn.operations.moreh.bmm(
        ttnn_input,
        ttnn_mat2,
        output=ttnn_output if optional_output else None,
        compute_kernel_config=compute_kernel_config,
    )
    actual_output = ttnn.to_torch(ttnn_output)

    # Compare results for equivalence
    rtol = atol = 0.1
    pcc = 0.998 if compute_kernel_options else 0.97
    passing, output_pcc = comp_allclose_and_pcc(output, actual_output, pcc=pcc, rtol=rtol, atol=atol)
    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")
    assert passing


def run_moreh_bmm_backward(
    shape,
    requires_grad,
    compute_kernel_options,
    device,
    *,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
):
    """
    Function to test the backward pass of Batched Matrix Multiplication (BMM) using PyTorch and TTNN backends.

    Args:
        shape (list): Shape of the input tensors [batch, m, k, n].
        requires_grad (tuple): Flags indicating whether gradients are required for input and mat2.
        compute_kernel_options: Configuration options for the compute kernel.
        device: The device on which to run the operations.
    """
    require_input_grad, require_mat2_grad = requires_grad
    input_shape = [shape[0], shape[1], shape[2]]
    mat2_shape = [shape[0], shape[2], shape[3]]
    output_shape = [shape[0], shape[1], shape[3]]
    (
        ttnn_input,
        ttnn_mat2,
        _,
        ttnn_output_grad,
        ttnn_input_grad,
        ttnn_mat2_grad,
        input,
        mat2,
        output_grad,
    ) = get_tensors(
        input_shape,
        mat2_shape,
        output_shape,
        require_input_grad,
        require_mat2_grad,
        device,
        torch_dtype=torch_dtype,
        ttnn_dtype=ttnn_dtype,
    )
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # Perform PyTorch BMM (backward)
    output = torch.bmm(input.requires_grad_(require_input_grad), mat2.requires_grad_(require_mat2_grad))
    output.backward(output_grad)

    # Perform TTNN BMM (backward)
    ttnn_input_grad, ttnn_mat2_grad = ttnn.operations.moreh.bmm_backward(
        ttnn_output_grad,
        ttnn_input,
        mat2=ttnn_mat2,
        are_required_outputs=(require_input_grad, require_mat2_grad),
        input_grad=ttnn_input_grad if require_input_grad else None,
        mat2_grad=ttnn_mat2_grad if require_mat2_grad else None,
        compute_kernel_config=compute_kernel_config,
    )

    # Compare results for equivalence
    rtol = atol = 0.1
    pcc = 0.998 if compute_kernel_options else 0.97
    if require_input_grad:
        actual_input_grad = ttnn.to_torch(ttnn_input_grad)
        passing, output_pcc = comp_allclose_and_pcc(input.grad, actual_input_grad, pcc=pcc, rtol=rtol, atol=atol)
        logger.info(f"input_grad passing={passing}")
        logger.info(f"input_grad pcc={output_pcc}")
        assert passing

    if require_mat2_grad:
        actual_mat2_grad = ttnn.to_torch(ttnn_mat2_grad)
        passing, output_pcc = comp_allclose_and_pcc(mat2.grad, actual_mat2_grad, pcc=pcc, rtol=rtol, atol=atol)
        logger.info(f"mat2_grad passing={passing}")
        logger.info(f"mat2_grad pcc={output_pcc}")
        assert passing


@pytest.mark.parametrize(
    "shape",
    [
        [1, 31, 639, 31],
        [5, 95, 415, 65],
        [10, 77, 320, 320],
        [10, 191, 447, 159],
    ],
)
def test_moreh_bmm_shape(shape, device):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm(shape, True, False if is_grayskull() else True, device)


@pytest.mark.parametrize("optional_output", [False, True])
def test_moreh_bmm_optional_output(optional_output, device):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm([10, 191, 447, 159], optional_output, False if is_grayskull() else True, device)


@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm_compute_kernel_options(compute_kernel_options, device):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm([10, 191, 447, 159], True, compute_kernel_options, device)


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_moreh_bmm_ttnn_dtype(ttnn_dtype, device):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    """
    # TODO @mrshaw01: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")
    torch.manual_seed(2024)
    run_moreh_bmm([10, 191, 447, 159], True, False if is_grayskull() else True, device, ttnn_dtype=ttnn_dtype)


@pytest.mark.parametrize(
    "shape",
    [[10, 191, 447, 159]],
)
def test_moreh_bmm_callback(shape, device, use_program_cache):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    AssertionError: If the number of program cache entries differs between runs with the same settings.
    """
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_bmm(shape, True, False if is_grayskull() else True, device)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 32, 32],
        [3, 31, 31, 31],
        [10, 77, 320, 320],
        [7, 511, 313, 765],
    ],
)
def test_moreh_bmm_backward_shape(shape, device):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm_backward(shape, [True, True], False if is_grayskull() else True, device)


@pytest.mark.parametrize(
    "requires_grad",
    [
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_moreh_bmm_backward_requires_grad(requires_grad, device):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm_backward([7, 511, 313, 765], requires_grad, False if is_grayskull() else True, device)


@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm_backward_compute_kernel_options(compute_kernel_options, device):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm_backward([7, 511, 313, 765], [True, True], compute_kernel_options, device)


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_moreh_bmm_backward_ttnn_dtype(ttnn_dtype, device):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    """
    # TODO @mrshaw01: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")
    torch.manual_seed(2024)
    run_moreh_bmm_backward(
        [7, 511, 313, 765], [True, True], False if is_grayskull() else True, device, ttnn_dtype=ttnn_dtype
    )


@pytest.mark.parametrize(
    "requires_grad",
    [
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_moreh_bmm_backward_callback(requires_grad, device, use_program_cache):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    AssertionError: If the number of program cache entries differs between runs with the same settings.
    """
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_bmm_backward([7, 511, 313, 765], requires_grad, False if is_grayskull() else True, device)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
