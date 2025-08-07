# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_ulp
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


def is_supported(shape, dim, ttnn_dtype):
    tensor_rank = len(shape)

    if dim < tensor_rank:
        accumulation_length = shape[dim]
        if ttnn_dtype == ttnn.bfloat16 and accumulation_length > 10000:
            return False  # for bfloat16, accmulation errors can happen easily on long tensor

    if ttnn_dtype == ttnn.float32 or ttnn_dtype == ttnn.bfloat16:
        return True

    if ttnn_dtype != ttnn.int32 and ttnn_dtype != ttnn.uint32:
        return False

    # For now, int32 version only supports >3-D tensors and `dim` outher than x and y axes
    if tensor_rank < 3:
        return False

    # int32/uin32: dim can not be x or y axes
    if dim == -1 or dim == -2 or dim == tensor_rank - 1 or dim == tensor_rank - 2:
        return False

    return True


@pytest.mark.parametrize(
    "size, dim",
    [
        # ([], 0),
        # ([0], 0),
        # ([1], 0),
        # ([10], 0),
        # ([2, 3], 0),
        # ([2, 3], 1),
        # ([2, 3], -1),
        # ([2, 3], -2),
        # ([2, 3, 4], 0),
        # ([2, 3, 4], 2),
        # ([2, 3, 4], -3),
        # ([0, 0, 0], 0),
        # ([0, 0, 0], 1),
        # ([1, 32, 64], 1),
        # ([1, 1024, 32], 0),
        # ([1, 1024, 32], 1),
        # ([260, 1, 1], 0),
        # ([1024, 1, 32], 0),
        # ([1, 1024, 32], 2),
        # ([64, 1, 32], 1),
        # ([64, 64, 1], 1),
        # ([1, 32, 129], 1),
        # ([33, 35, 37], 1),
        ([1, 12, 40, 256], 1),
        ([1, 12, 40, 256], 2),
        ([1, 24, 80, 256], 1),
        ([1, 24, 80, 256], 2),
        ([1, 48, 160, 256], 1),
        ([1, 48, 160, 256], 2),
        # ([7, 13, 129, 33], 1),
        # ([7, 13, 129, 33], 0),
        # ([4, 6, 128, 128], 0),
        # ([2, 3, 5, 33, 128], 0),
        # ([2, 3, 5, 33, 128], 1),
        # ([2, 3, 5, 33, 128], 2),
        # ([2, 3, 5, 33, 128], 3),
        # ([2, 3, 5, 33, 128], 4),
        # ((2, 3, 4, 5, 33, 33), 0),
        # ((2, 3, 4, 5, 33, 33), 1),
        # ((2, 3, 4, 5, 33, 33), 2),
        # ((2, 3, 4, 5, 33, 33), 3),
        # ((2, 3, 4, 5, 33, 33), 4),
        # ((2, 3, 4, 5, 33, 33), 5),
        # ([8192, 16, 32, 32], 1),
        # ([1, 151936], -1),
        # ([1, 19], -1),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        # (torch.float32, None),
        (torch.bfloat16, ttnn.bfloat16),
        # (torch.float32, ttnn.float32),
        # (torch.float32, ttnn.bfloat16),
        # (torch.int32, ttnn.int32),
    ],
)
def test_cumsum(size, dim, dtypes, device):
    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    # Generate integer input on [-2; 2];
    # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
    # which are not handled yet
    # torch_input_tensor = torch.randint(-2, 3, size=size, dtype=torch_dtype)
    torch_input_tensor = torch.rand(size=size, dtype=torch_dtype)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.Layout.TILE, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.Layout.TILE)

    expected_output_dtype = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    # For now, int32 version only supports >3-D tensors and `dim` outher than x and y axes
    if not is_supported(size, dim, expected_output_dtype):
        pytest.skip("Unsupported configuration by ttnn.cumsum")

    output_tensor = ttnn.cumsum(input_tensor, dim=dim, dtype=ttnn_dtype)

    assert output_tensor.dtype == expected_output_dtype
    assert output_tensor.shape == (size)

    torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    expected_output = torch.cumsum(torch_input_tensor, dim=dim, dtype=torch_dtype)

    comp_allclose_and_pcc(expected_output, torch_output)
    if torch_output.numel() > 0:
        assert_allclose(expected_output, torch_output, atol=4.0, rtol=1e-3)


@pytest.mark.parametrize(
    "size",
    [
        ([1, 12, 40, 256]),
        ([1, 24, 80, 256]),
        ([1, 48, 160, 256]),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.L1_MEMORY_CONFIG,
        # ttnn.DRAM_MEMORY_CONFIG,
    ],
)
def test_integral_image(size, dtypes, memory_config, device):
    torch.manual_seed(29112024)

    def integral_image(features):
        return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)

    def integral_image_channel_last(features):
        assert len(features.shape) == 4, "Input tensor must be 4D"
        assert features.shape[0] == 1, "Batch size must be 1"
        return ttnn.cumsum(ttnn.cumsum(features, dim=1, dtype=features.dtype), dim=2, dtype=features.dtype)

    (torch_dtype, ttnn_dtype) = dtypes

    # Generate integer input on [-2; 2];
    # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
    # which are not handled yet
    # torch_input_tensor = torch.randint(-2, 3, size=size, dtype=torch_dtype)
    torch_input_tensor = torch.rand(size=size, dtype=torch_dtype)
    # input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.Layout.TILE, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.Layout.TILE, memory_config=memory_config
    )

    expected_output_dtype = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    # For now, int32 version only supports >3-D tensors and `dim` outher than x and y axes
    # if not is_supported(size, dim, expected_output_dtype):
    #     pytest.skip("Unsupported configuration by ttnn.cumsum")

    output_tensor = integral_image_channel_last(input_tensor)

    assert output_tensor.dtype == expected_output_dtype
    assert output_tensor.shape == (size)

    torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    torch_input_channel_first = torch.permute(torch_input_tensor, (0, 3, 1, 2))
    expected_output = integral_image(torch_input_channel_first)
    expected_output = torch.permute(expected_output, (0, 2, 3, 1))

    comp_allclose_and_pcc(expected_output, torch_output)
    if torch_output.numel() > 0:
        assert_allclose(expected_output, torch_output, atol=4.0, rtol=1e-3)


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([1], 0),
        ([2, 3], 0),
        ([2, 3], -1),
        ([1, 1024, 32], 0),
        ([33, 35, 37], -1),
        ([7, 13, 129, 33], 1),
        ([2, 3, 5, 33, 128], -1),
        ([5, 2, 3, 5, 33, 128], 0),
        ([1, 151936], -1),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.float32, ttnn.bfloat16),
    ],
)
def test_cumsum_with_preallocated_output(size, dim, dtypes, device):
    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    torch_input_tensor = torch.randint(-2, 3, size, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn_dtype, layout=ttnn.Layout.TILE)

    expected_output_dtype = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    # For now, test_cumsum_with_preallocated_output ony support bfloat16 and float32
    if expected_output_dtype == ttnn.int32 or expected_output_dtype == ttnn.uint32:
        pytest.skip("ttnn.cumsum with preallocated output does not support integer types")

    if not is_supported(size, dim, expected_output_dtype):
        pytest.skip("Unsupported configuration by ttnn.cumsum")

    preallocated_output_tensor = ttnn.zeros_like(input_tensor, dtype=ttnn_dtype, layout=ttnn.Layout.TILE)

    output_tensor = ttnn.cumsum(input_tensor, dim=dim, dtype=ttnn_dtype, out=preallocated_output_tensor)
    torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    expected_output = torch.cumsum(torch_input_tensor, dim=dim, dtype=torch_dtype)

    assert output_tensor.dtype == expected_output_dtype
    assert preallocated_output_tensor.dtype == expected_output_dtype

    assert output_tensor.shape == (size)
    assert preallocated_output_tensor.shape == (size)

    assert preallocated_output_tensor == output_tensor

    if torch_output.numel() > 0:
        if torch_dtype is torch.float32:
            assert_allclose(expected_output, torch_output, atol=4.0, rtol=1e-3)
        else:
            assert_allclose(expected_output, torch_output)

    assert device.num_program_cache_entries() >= 1


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([1], 0),
        ([2, 3], 0),
        ([2, 3], -1),
        ([1, 1024, 32], 0),
        ([33, 35, 37], -1),
        ([7, 13, 129, 33], 1),
        ([2, 3, 5, 33, 128], -1),
        ([5, 2, 3, 5, 33, 128], 0),
        ([1, 151936], -1),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
    ],
)
def test_cumsum_backward(size, dim, dtypes, device):
    output_shape = size.copy()

    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    # Generate integer input on [-2; 2];
    # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
    # which are not handled yet
    torch_input_tensor = torch.randint(-2, 3, size=size, dtype=torch_dtype, requires_grad=True)
    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.Layout.TILE)

    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(size, size, device)

    torch_output = torch.cumsum(torch_input_tensor, dim)
    torch_output.backward(torch_output_grad)

    tt_input_grad_cpu = ttnn.to_torch(
        ttnn.cumsum(tt_output_grad, dim, dtype=ttnn_dtype, reverse_order=True, out=tt_input_grad)
    )

    assert tt_input_grad_cpu.shape == torch_input_tensor.grad.shape

    # test for equivalance
    rtol = atol = 0.1
    assert comp_allclose_and_pcc(torch_input_tensor.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)


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
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.ROW_MAJOR,
        ),  # unsupported layout
    ],
)
def test_cumsum_failing_cases(
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
        ttnn.cumsum(ttnn_input_tensor, memory_config=memory_config, dim=dim, out=ttnn_preallocated_tensor)
