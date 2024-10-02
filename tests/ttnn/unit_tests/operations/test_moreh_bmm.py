# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.test_moreh_matmul import get_tensors
from models.utility_functions import comp_allclose_and_pcc

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_npu,
)


def run_moreh_bmm(shape, optional_output, compute_kernel_options, device):
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
    ) = get_tensors(input_shape, mat2_shape, output_shape, False, False, False, device)
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # Perform PyTorch BMM
    output = torch.bmm(input, mat2)

    # Perform TTNN BMM
    ttnn_output = (
        ttnn.operations.moreh.bmm(
            ttnn_input,
            ttnn_mat2,
            output=ttnn_output if optional_output else None,
            compute_kernel_config=compute_kernel_config,
        )
        .cpu()
        .to(ttnn.ROW_MAJOR_LAYOUT)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # Compare results for equivalence
    passing, output_pcc = comp_allclose_and_pcc(output, ttnn_output, pcc=0.999)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing


def run_moreh_bmm_backward(shape, requires_grad, compute_kernel_options, device):
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
    ) = get_tensors(input_shape, mat2_shape, output_shape, require_input_grad, require_mat2_grad, False, device)
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
    if require_input_grad:
        ttnn_cpu_input_grad = ttnn_input_grad.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(input.grad, ttnn_cpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_mat2_grad:
        ttnn_cpu_mat2_grad = ttnn_mat2_grad.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(mat2_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(mat2.grad, ttnn_cpu_mat2_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"mat2_grad passing={passing}")
        logger.debug(f"mat2_grad pcc={output_pcc}")
        assert passing


@pytest.mark.parametrize(
    "shape",
    [
        [1, 31, 639, 31],
        [5, 95, 415, 65],
        [10, 191, 447, 159],
    ],
)
@pytest.mark.parametrize("optional_output", [False, True])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm(shape, optional_output, compute_kernel_options, device):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm(shape, optional_output, compute_kernel_options, device)


@pytest.mark.parametrize(
    "shape",
    [[10, 191, 447, 159]],
)
@pytest.mark.parametrize("optional_output", [False, True])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm_callback(shape, optional_output, compute_kernel_options, device, use_program_cache):
    """
    PyTest wrapper for running BMM tests with multiple configurations.
    AssertionError: If the number of program cache entries differs between runs with the same settings.
    """
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_bmm(shape, optional_output, compute_kernel_options, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 32, 32],
        [3, 31, 31, 31],
        [7, 511, 313, 765],
    ],
)
@pytest.mark.parametrize(
    "requires_grad",
    [
        [True, False],
        [False, True],
        [True, True],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm_backward(shape, requires_grad, compute_kernel_options, device):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    """
    torch.manual_seed(2024)
    run_moreh_bmm_backward(shape, requires_grad, compute_kernel_options, device)


@pytest.mark.parametrize(
    "shape",
    [[7, 511, 313, 765]],
)
@pytest.mark.parametrize(
    "requires_grad",
    [
        [True, False],
        [False, True],
        [True, True],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm_backward_callback(shape, requires_grad, compute_kernel_options, device, use_program_cache):
    """
    PyTest wrapper for running BMM backward tests with multiple configurations.
    AssertionError: If the number of program cache entries differs between runs with the same settings.
    """
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_bmm_backward(shape, requires_grad, compute_kernel_options, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
