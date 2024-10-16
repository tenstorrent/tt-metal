# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.utility_functions import comp_allclose, is_wormhole_b0
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_torch,
    to_ttnn,
    compute_output_shape,
    TILE_HEIGHT,
    TILE_WIDTH,
    check_dim,
)


def make_cpu_tensors(input_shape, dim, keepdim=False):
    """
    Creates random CPU tensors for input and gradient output based on the input shape and dimension.

    Args:
        input_shape (tuple): The shape of the input tensor.
        dim (int or tuple of int, optional): Dimension(s) over which to compute the norm.
        keepdim (bool, optional): Whether to keep the dimensions of the output. Defaults to False.

    Returns:
        tuple: A tuple containing two tensors:
            - `cpu_input`: Random input tensor on CPU with `requires_grad=True`.
            - `cpu_output_grad`: Random gradient tensor with the shape corresponding to the output.
    """
    torch_output_shape, _ = compute_output_shape(input_shape, dim, keepdim=keepdim)
    cpu_input = torch.empty(input_shape, dtype=torch.float32).uniform_(-1, 1).requires_grad_()
    cpu_output_grad = torch.empty(torch_output_shape, dtype=torch.float32).uniform_(-1, 1)
    return cpu_input, cpu_output_grad


def torch_norm(cpu_x, cpu_dy, *, p=2.0, dim=None, keepdim=False, do_backward=False):
    """
    Computes the norm of a tensor using PyTorch and optionally performs backpropagation.

    Args:
        cpu_x (torch.Tensor): Input tensor.
        cpu_dy (torch.Tensor): Gradient tensor for backpropagation.
        p (float, optional): The order of the norm. Defaults to 2.0.
        dim (int or tuple of int, optional): Dimension(s) over which to compute the norm.
        keepdim (bool, optional): Whether to keep the dimensions of the output. Defaults to False.
        do_backward (bool, optional): If True, performs backpropagation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - `cpu_y`: The result of the norm operation.
            - `cpu_dx`: The gradient of the input tensor (if `do_backward=True`), otherwise None.
    """
    cpu_y = torch.norm(cpu_x, p=p, dim=dim, keepdim=keepdim)
    cpu_dx = None
    if do_backward:
        cpu_y.backward(cpu_dy)
        cpu_dx = cpu_x.grad
    return cpu_y, cpu_dx


def tt_norm(
    cpu_x,
    cpu_dy,
    *,
    p=2.0,
    dim=None,
    keepdim=False,
    compute_kernel_options=None,
    do_backward=False,
    device=None,
):
    """
    Computes the norm of a tensor using Tenstorrent's custom backend and optionally performs backpropagation.

    Args:
        cpu_x (torch.Tensor): Input tensor on CPU.
        cpu_dy (torch.Tensor): Gradient tensor for backpropagation on CPU.
        p (float, optional): The order of the norm. Defaults to 2.0.
        dim (int or tuple of int, optional): Dimension(s) over which to compute the norm.
        keepdim (bool, optional): Whether to keep the dimensions of the output. Defaults to False.
        compute_kernel_options: Configuration options for the compute kernel.
        do_backward (bool, optional): If True, performs backpropagation. Defaults to False.
        device (torch.device, optional): The device to run the computation on. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - `npu_y`: The result of the norm operation.
            - `npu_dx`: The gradient of the input tensor (if `do_backward=True`), otherwise None.
    """
    _, tt_output_shape = compute_output_shape(cpu_x.shape, dim, keepdim=keepdim)
    npu_x = to_ttnn(cpu_x.bfloat16(), device=device)
    if do_backward:
        npu_dy = to_ttnn(cpu_dy.reshape(tt_output_shape).bfloat16(), device=device)
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    if do_backward:
        npu_y = to_ttnn(
            torch.norm(cpu_x, p=p, dim=dim, keepdim=keepdim).bfloat16().reshape(tt_output_shape), device=device
        )
    else:
        npu_y = to_ttnn(torch.empty(tt_output_shape), device=device)
        ttnn.operations.moreh.norm(
            npu_x, p=p, dim=dim, keepdim=keepdim, output=npu_y, compute_kernel_config=compute_kernel_config
        )

    npu_dx = None
    if do_backward:
        npu_dx = to_ttnn(torch.empty_like(cpu_x), device=device)

        ttnn.operations.moreh.norm_backward(
            npu_x,
            npu_y,
            npu_dy,
            p=p,
            dim=dim,
            keepdim=keepdim,
            input_grad=npu_dx,
            compute_kernel_config=compute_kernel_config,
        )
        npu_dx = to_torch(npu_dx, shape=cpu_x.shape)

    npu_y = to_torch(npu_y, shape=tt_output_shape)
    return npu_y, npu_dx


def run_moreh_norm(input_shape, p, dim, rtol, atol, device, keepdim=False, compute_kernel_options=None):
    """
    Runs the norm operation using both PyTorch and Tenstorrent's custom implementation and compares the outputs.

    Args:
        input_shape (tuple): The shape of the input tensor.
        p (float): The order of the norm.
        dim (int or list of int, optional): Dimension(s) over which to compute the norm.
        rtol (float): Relative tolerance for the comparison.
        atol (float): Absolute tolerance for the comparison.
        device: The device to run the computation on.
        keepdim (bool, optional): Whether to retain the reduced dimensions in the output. Defaults to False.
        compute_kernel_options: Configuration options for the compute kernel.

    Raises:
        AssertionError: If the computed norm values from Tenstorrent's implementation and PyTorch are not close.
    """
    if dim in (None, [], [0, 1, 2, 3]) and p == 2.5 and is_wormhole_b0():
        pytest.skip("TODO: Check why comp_allclose result is poor on WH_B0.")
    check_dim(input_shape, dim, keepdim)
    cpu_x, cpu_dy = make_cpu_tensors(input_shape, dim, keepdim=keepdim)

    expected_y, _ = torch_norm(cpu_x, cpu_dy, p=p, dim=dim, keepdim=keepdim, do_backward=False)
    actual_y, _ = tt_norm(
        cpu_x,
        cpu_dy,
        p=p,
        dim=dim,
        keepdim=keepdim,
        compute_kernel_options=compute_kernel_options,
        device=device,
        do_backward=False,
    )
    actual_y = actual_y if keepdim else actual_y.reshape(expected_y.shape)

    pass_y, out_y = comp_allclose(expected_y, actual_y, rtol=rtol, atol=atol)
    logger.debug(f"output's {out_y}")
    assert pass_y


def run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, keepdim=False, compute_kernel_options=None):
    """
    Runs the norm operation with backpropagation using both PyTorch and Tenstorrent's custom implementation and compares the gradients.

    Args:
        input_shape (tuple): The shape of the input tensor.
        p (float): The order of the norm.
        dim (int or list of int, optional): Dimension(s) over which to compute the norm.
        rtol (float): Relative tolerance for the comparison.
        atol (float): Absolute tolerance for the comparison.
        device: The device to run the computation on.
        keepdim (bool, optional): Whether to retain the reduced dimensions in the output. Defaults to False.
        compute_kernel_options: Configuration options for the compute kernel.

    Raises:
        AssertionError: If the computed gradients from Tenstorrent's implementation and PyTorch are not close.
    """
    check_dim(input_shape, dim, keepdim)

    cpu_x, cpu_dy = make_cpu_tensors(input_shape, dim, keepdim=keepdim)

    _, expected_dx = torch_norm(cpu_x, cpu_dy, p=p, dim=dim, keepdim=keepdim, do_backward=True)
    _, actual_dx = tt_norm(
        cpu_x,
        cpu_dy,
        p=p,
        dim=dim,
        keepdim=keepdim,
        compute_kernel_options=compute_kernel_options,
        device=device,
        do_backward=True,
    )

    pass_dx, out_dx = comp_allclose(expected_dx, actual_dx, rtol=rtol, atol=atol)
    logger.debug(f"input_grad's {out_dx}")
    assert pass_dx


@pytest.mark.parametrize("p", [2.0, 2.5, -2.5])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[], 0.2, 0.2],
        [None, 0.2, 0.2],
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
        [3, 0.1, 0.1],
        [[0, 1], 0.1, 0.1],
        [[0, 1, 2], 0.15, 0.15],
        [[0, 1, 2, 3], 0.2, 0.2],
        [[0, 1, 3], 0.15, 0.15],
        [[0, 2, 3], 0.15, 0.15],
        [[1, 2], 0.1, 0.1],
        [[1, 2, 3], 0.15, 0.15],
        [[1, 3], 0.1, 0.1],
        [[2, 3], 0.1, 0.1],
    ],
    ids=[
        "global_norm(dim=[])",
        "global_norm(dim=None)",
        "N",
        "C",
        "H",
        "W",
        "NC",
        "NCH",
        "NCHW",
        "NCW",
        "NHW",
        "CH",
        "CHW",
        "CW",
        "HW",
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_norm(input_shape, p, dim_rtol_atol, keepdim, device):
    """
    Parametrized test for Tenstorrent's norm operation. Compares the output of Tenstorrent's norm with PyTorch's norm
    for various input shapes, dimensions, norms, and keepdim settings.

    Args:
        input_shape (list of int): Shape of the input tensor.
        p (float): The order of the norm.
        dim_rtol_atol (list): List containing the dimension(s), relative tolerance, and absolute tolerance for the comparison.
        keepdim (bool): Whether to retain the reduced dimensions in the output.
        device: The device to run the computation on.

    Raises:
        AssertionError: If the computed outputs from Tenstorrent's implementation and PyTorch are not close.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm(input_shape, p, dim, rtol, atol, device, keepdim=keepdim)


@pytest.mark.parametrize("p", [2.0, 2.5, -2.5])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [10, TILE_HEIGHT, TILE_WIDTH],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_norm_compute_kernel_options(input_shape, p, dim_rtol_atol, compute_kernel_options, device):
    """
    Parametrized test for Tenstorrent's norm operation with various kernel compute options. Compares the output of Tenstorrent's norm
    with PyTorch's norm for different input shapes, dimensions, norms, and compute kernel configurations.

    Args:
        input_shape (list of int): Shape of the input tensor.
        p (float): The order of the norm.
        dim_rtol_atol (list): List containing the dimension(s), relative tolerance, and absolute tolerance for the comparison.
        compute_kernel_options: Configuration options for the compute kernel.
        device: The device to run the computation on.

    Raises:
        AssertionError: If the computed outputs from Tenstorrent's implementation and PyTorch are not close.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm(input_shape, p, dim, rtol, atol, device, compute_kernel_options=compute_kernel_options)


@pytest.mark.parametrize("p", [2.0])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[0, 1, 2, 3], 0.2, 0.2],
    ],
    ids=[
        "NCHW",
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_norm_callback(input_shape, p, dim_rtol_atol, keepdim, device, use_program_cache):
    """
    Test the norm operation in Tenstorrent's implementation with and without the program cache.
    Verifies that the number of program cache entries remains consistent when running norm operations.

    Args:
        input_shape (list of int): Shape of the input tensor.
        p (float): The order of the norm.
        dim_rtol_atol (list): List containing the dimension(s), relative tolerance, and absolute tolerance for the comparison.
        device: The device to run the computation on.
        use_program_cache: Use the program cache.

    Raises:
        AssertionError: If the number of program cache entries differs between runs with the same settings.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_norm(input_shape, p, dim, rtol, atol, device, keepdim=keepdim)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize("p", [2.0])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[], 0.2, 0.2],
        [None, 0.2, 0.2],
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
        [3, 0.1, 0.1],
        [[0, 1], 0.1, 0.1],
        [[0, 1, 2], 0.15, 0.15],
        [[0, 1, 2, 3], 0.2, 0.2],
        [[0, 1, 3], 0.15, 0.15],
        [[0, 2, 3], 0.15, 0.15],
        [[1, 2], 0.1, 0.1],
        [[1, 2, 3], 0.15, 0.15],
        [[1, 3], 0.1, 0.1],
        [[2, 3], 0.1, 0.1],
    ],
    ids=[
        "global_norm(dim=[])",
        "global_norm(dim=None)",
        "N",
        "C",
        "H",
        "W",
        "NC",
        "NCH",
        "NCHW",
        "NCW",
        "NHW",
        "CH",
        "CHW",
        "CW",
        "HW",
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_norm_backward(input_shape, p, dim_rtol_atol, keepdim, device):
    """
    Parametrized test for Tenstorrent's norm operation with backward propagation.
    Compares the output gradient of Tenstorrent's norm with PyTorch's norm across various configurations.

    Args:
        input_shape (list of int): Shape of the input tensor.
        p (float): The order of the norm.
        dim_rtol_atol (list): List containing the dimension(s), relative tolerance, and absolute tolerance for the comparison.
        keepdim (bool): Whether to retain the reduced dimensions in the output.
        device: The device to run the computation on.

    Raises:
        AssertionError: If the computed gradients from Tenstorrent's implementation and PyTorch are not close.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, keepdim=keepdim)


@pytest.mark.parametrize("p", [2.0, 2.5, -2.5])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [10, TILE_HEIGHT, TILE_WIDTH],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_norm_backward_compute_kernel_options(input_shape, p, dim_rtol_atol, compute_kernel_options, device):
    """
    Parametrized test for Tenstorrent's norm backward operation with various kernel compute options.
    Compares the output gradient of Tenstorrent's norm with PyTorch's norm across different compute kernel configurations.

    Args:
        input_shape (list of int): Shape of the input tensor.
        p (float): The order of the norm.
        dim_rtol_atol (list): List containing the dimension(s), relative tolerance, and absolute tolerance for the comparison.
        compute_kernel_options: Configuration options for the compute kernel.
        device: The device to run the computation on.

    Raises:
        AssertionError: If the computed gradients from Tenstorrent's implementation and PyTorch are not close.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, compute_kernel_options=compute_kernel_options)


@pytest.mark.parametrize("p", [2.0])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[0, 1, 2, 3], 0.2, 0.2],
    ],
    ids=[
        "NCHW",
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_norm_backward_callback(input_shape, p, dim_rtol_atol, keepdim, device, use_program_cache):
    """
    Test the norm backward operation in Tenstorrent's implementation with and without the program cache.
    Verifies that the number of program cache entries remains consistent when running backward norm operations.

    Args:
        input_shape (list of int): Shape of the input tensor.
        p (float): The order of the norm.
        dim_rtol_atol (list): List containing the dimension(s), relative tolerance, and absolute tolerance for the comparison.
        device: The device to run the computation on.
        use_program_cache: Use the program cache.

    Raises:
        AssertionError: If the number of program cache entries differs between runs with the same settings.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, keepdim=keepdim)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
