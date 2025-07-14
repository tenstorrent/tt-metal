# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_allclose_and_pcc


def get_backward_tensors(output_grad_shape, input_grad_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype, requires_grad=True)
    torch_input_grad = torch.randint(-2, 3, input_grad_shape, dtype=cpu_dtype)

    tt_output_grad = ttnn.Tensor(torch_output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_input_grad = ttnn.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [2],
        [2000],
        [1000, 32, 32],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_normal(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=dtypes[0])
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
            )
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, dtype=dtypes[1])

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            # assert values with pcc
            assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, 0.99)
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [2],
        [2000],
        [1000, 32, 32],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_backward(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        (torch_dtype, ttnn_dtype) = dtypes

        # Generate integer input on [-2; 2];
        # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
        # which are not handled yet
        torch_input_tensor = torch.randint(-2, 3, size=shape, dtype=torch_dtype, requires_grad=True)

        (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(shape, shape, device)

        torch_output = torch.cumsum(torch_input_tensor, dim)
        torch_output.backward(torch_output_grad)

        tt_input_grad_cpu = ttnn.to_torch(
            ttnn.cumsum(tt_output_grad, dim, dtype=ttnn_dtype, reverse_order=True, input_grad=tt_input_grad)
        )

        assert tt_input_grad_cpu.shape == torch_input_tensor.grad.shape

        # test for equivalance
        rtol = atol = 0.1
        passing, _ = comp_allclose_and_pcc(torch_input_tensor.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)

        assert passing
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [1],
        [2000],
        [1000, 32, 32],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
    ],
)
def test_cumprod_preallocated(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=dtypes[0])
            torch_preallocated_tensor = torch.zeros_like(torch_input_tensor)
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim, out=torch_preallocated_tensor)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, dtype=dtypes[1], layout=ttnn.Layout.TILE, device=device
            )
            ttnn_preallocated_tensor = ttnn.zeros_like(ttnn_input_tensor)
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, out=ttnn_preallocated_tensor)

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.dtype == ttnn_result_tensor.dtype
            assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            # assert values with pcc
            comp_allclose_and_pcc()
            assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, 0.99)
            assert_with_pcc(ttnn.to_torch(ttnn_preallocated_tensor), torch_preallocated_tensor, 0.98)
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


@pytest.mark.parametrize(
    "dim, input_shape, output_shape, torch_dtype, input_dtype, output_dtype, memory_config, layout",
    [
        (
            -10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_rank vs dim
        (
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_rank vs dim
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 1],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.float32,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_dtype vs output_dtype
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # unsupported memory config
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.ROW_MAJOR,
        ),  # unsupported layout
    ],
)
def test_cumprod_failing_cases(
    dim,
    input_shape,
    output_shape,
    torch_dtype,
    input_dtype,
    output_dtype,
    memory_config,
    layout,
    device,
):
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=layout, device=device, memory_config=memory_config
    )
    ttnn_preallocated_tensor = ttnn.zeros(output_shape, dtype=output_dtype)
    with pytest.raises(RuntimeError):
        ttnn.cumprod(ttnn_input_tensor, dim=dim, out=ttnn_preallocated_tensor)
