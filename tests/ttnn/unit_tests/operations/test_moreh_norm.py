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
    compute_output_shape,
    check_dim,
)


def create_ttnn_tilized_tensor(torch_tensor, device, dtype):
    return ttnn.from_torch(torch_tensor, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def make_torch_tensors(input_shape, dim, keepdim=False, *, dtype=torch.float32):
    """
    Creates random tensors for input and gradient output based on the input shape and dimension.

    Args:
        input_shape (tuple): The shape of the input tensor.
        dim (int or tuple of int, optional): Dimension(s) over which to compute the norm.
        keepdim (bool, optional): Whether to keep the dimensions of the output. Defaults to False.

    Returns:
        tuple: A tuple containing two tensors:
            - `torch_input`: Random input tensor on CPU with `requires_grad=True`.
            - `torch_output_grad`: Random gradient tensor with the shape corresponding to the output.
    """
    torch_output_shape, _ = compute_output_shape(input_shape, dim, keepdim=keepdim)
    torch_input = torch.empty(input_shape, dtype=dtype).uniform_(-1, 1).requires_grad_()
    torch_output_grad = torch.empty(torch_output_shape, dtype=dtype).uniform_(-1, 1)
    return torch_input, torch_output_grad


def torch_norm(torch_input, torch_output_grad, *, p=2.0, dim=None, keepdim=False, do_backward=False):
    """
    Computes the norm of a tensor using torch and optionally performs backpropagation.

    Args:
        torch_input (torch.Tensor): Input tensor.
        torch_output_grad (torch.Tensor): Gradient output tensor for backpropagation.
        p (float, optional): The order of the norm. Defaults to 2.0.
        dim (int or tuple of int, optional): Dimension(s) over which to compute the norm.
        keepdim (bool, optional): Whether to keep the dimensions of the output. Defaults to False.
        do_backward (bool, optional): If True, performs backpropagation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - `torch_output`: The result of the norm operation.
            - `torch_input_grad`: The gradient of the input tensor (if `do_backward=True`), otherwise None.
    """
    torch_output = torch.norm(torch_input, p=p, dim=dim, keepdim=keepdim)
    torch_input_grad = None
    if do_backward:
        torch_output.backward(torch_output_grad)
        torch_input_grad = torch_input.grad
    return torch_output, torch_input_grad


def ttnn_norm(
    torch_input,
    torch_output_grad,
    *,
    p=2.0,
    dim=None,
    keepdim=False,
    compute_kernel_options=None,
    do_backward=False,
    device=None,
    dtype=ttnn.bfloat16,
):
    """
    Computes the norm of a tensor using ttnn's custom backend and optionally performs backpropagation.

    Args:
        torch_input (torch.Tensor): Input tensor.
        torch_output_grad (torch.Tensor): Gradient output tensor for backpropagation.
        p (float, optional): The order of the norm. Defaults to 2.0.
        dim (int or tuple of int, optional): Dimension(s) over which to compute the norm.
        keepdim (bool, optional): Whether to keep the dimensions of the output. Defaults to False.
        compute_kernel_options: Configuration options for the compute kernel.
        do_backward (bool, optional): If True, performs backpropagation. Defaults to False.
        device (torch.device, optional): The device to run the computation on. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - `ttnn_output`: The result of the norm operation.
            - `ttnn_input_grad`: The gradient of the input tensor (if `do_backward=True`), otherwise None.
    """
    _, ttnn_output_shape = compute_output_shape(torch_input.shape, dim, keepdim=keepdim)
    ttnn_input = create_ttnn_tilized_tensor(torch_input, device, dtype)
    if do_backward:
        torch_output = torch.norm(torch_input, p=p, dim=dim, keepdim=keepdim)
        ttnn_output = create_ttnn_tilized_tensor(torch_output, device, dtype)
    else:
        ttnn_output = create_ttnn_tilized_tensor(torch.empty(ttnn_output_shape), device, dtype)
        ttnn.operations.moreh.norm(
            ttnn_input,
            p=p,
            dim=dim,
            keepdim=keepdim,
            output=ttnn_output,
            compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
        )
    ttnn_input_grad = None
    if do_backward:
        ttnn_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, dtype)
        ttnn_input_grad = create_ttnn_tilized_tensor(torch.empty_like(torch_input), device, dtype)
        ttnn.operations.moreh.norm_backward(
            ttnn_input,
            ttnn_output,
            ttnn_output_grad,
            p=p,
            dim=dim,
            keepdim=keepdim,
            input_grad=ttnn_input_grad,
            compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
        )
        ttnn_input_grad = ttnn.to_torch(ttnn_input_grad)
    ttnn_output = ttnn.to_torch(ttnn_output)
    return ttnn_output, ttnn_input_grad


def run_moreh_norm(
    input_shape,
    p,
    dim,
    rtol,
    atol,
    device,
    *,
    keepdim=False,
    compute_kernel_options=None,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
):
    """
    Runs the norm operation using both torch and ttnn's implementation and compares the outputs.

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
        AssertionError: If the computed norm values from ttnn's implementation and torch are not close.
    """
    # TODO @mrshaw01: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")
    check_dim(input_shape, dim, keepdim)
    torch_input, torch_output_grad = make_torch_tensors(input_shape, dim, keepdim=keepdim, dtype=torch_dtype)
    expected_output, _ = torch_norm(torch_input, torch_output_grad, p=p, dim=dim, keepdim=keepdim, do_backward=False)
    actual_output, _ = ttnn_norm(
        torch_input,
        torch_output_grad,
        p=p,
        dim=dim,
        keepdim=keepdim,
        compute_kernel_options=compute_kernel_options,
        device=device,
        do_backward=False,
        dtype=ttnn_dtype,
    )
    passing, out = comp_allclose(expected_output, actual_output, rtol=rtol, atol=atol)
    logger.info(f"output's {out}")
    assert passing


def run_moreh_norm_backward(
    input_shape,
    p,
    dim,
    rtol,
    atol,
    device,
    keepdim=False,
    compute_kernel_options=None,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
):
    """
    Runs the norm operation with backpropagation using both torch and ttnn's custom implementation and compares the gradients.

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
        AssertionError: If the computed gradients from ttnn's implementation and torch are not close.
    """
    # TODO @mrshaw01: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")
    check_dim(input_shape, dim, keepdim)
    torch_input, torch_output_grad = make_torch_tensors(input_shape, dim, keepdim=keepdim, dtype=torch_dtype)
    _, expected_input_grad = torch_norm(torch_input, torch_output_grad, p=p, dim=dim, keepdim=keepdim, do_backward=True)
    _, actual_input_grad = ttnn_norm(
        torch_input,
        torch_output_grad,
        p=p,
        dim=dim,
        keepdim=keepdim,
        compute_kernel_options=compute_kernel_options,
        device=device,
        do_backward=True,
        dtype=ttnn_dtype,
    )
    passing, out = comp_allclose(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol)
    logger.info(f"input_grad's {out}")
    assert passing


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
        [32, 32],
        [5, 8, 78, 77],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_norm(input_shape, p, dim_rtol_atol, keepdim, ttnn_dtype, device):
    """
    Parametrized test for ttnn's norm operation. Compares the output of ttnn's norm with torch's norm.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm(input_shape, p, dim, rtol, atol, device, keepdim=keepdim, ttnn_dtype=ttnn_dtype)


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
        [10, 32, 32],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_norm_compute_kernel_options(input_shape, p, dim_rtol_atol, compute_kernel_options, device):
    """
    Parametrized test for ttnn's norm operation. Compares the output of ttnn's norm with torch's norm.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm(
        input_shape,
        p,
        dim,
        rtol,
        atol,
        device,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[1, 3], 0.1, 0.1],
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
        [3, 0.1, 0.1],
    ],
    ids=["CW", "N", "C", "H", "W"],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_norm_callback(dim_rtol_atol, keepdim, device, use_program_cache):
    """
    Parametrized test for ttnn's norm operation. Compares the output of ttnn's norm with torch's norm.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_norm([5, 8, 78, 77], 2.0, dim, rtol, atol, device, keepdim=keepdim)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


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
        [32, 32],
        [5, 8, 78, 77],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_norm_backward(input_shape, p, dim_rtol_atol, keepdim, ttnn_dtype, device):
    """
    Parametrized test for ttnn's norm backward operation. Compares the output of ttnn's norm backward with torch's norm backward.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, keepdim=keepdim, ttnn_dtype=ttnn_dtype)


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
        [10, 32, 32],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_norm_backward_compute_kernel_options(input_shape, p, dim_rtol_atol, compute_kernel_options, device):
    """
    Parametrized test for ttnn's norm backward operation. Compares the output of ttnn's norm backward with torch's norm backward.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    run_moreh_norm_backward(
        input_shape,
        p,
        dim,
        rtol,
        atol,
        device,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[1, 3], 0.1, 0.1],
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
        [3, 0.1, 0.1],
    ],
    ids=["CW", "N", "C", "H", "W"],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_moreh_norm_backward_callback(dim_rtol_atol, keepdim, device, use_program_cache):
    """
    Parametrized test for ttnn's norm backward operation. Compares the output of ttnn's norm backward with torch's norm backward.
    """
    torch.manual_seed(2024)
    dim, rtol, atol = dim_rtol_atol
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_norm_backward([5, 8, 78, 77], 2.0, dim, rtol, atol, device, keepdim=keepdim)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
