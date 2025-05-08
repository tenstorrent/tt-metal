# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def is_supported(tensor_rank, dim, ttnn_dtype):
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
        ([], 0),
        ([0], 0),
        ([1], 0),
        ([10], 0),
        ([2, 3], 0),
        ([2, 3], 1),
        ([2, 3], -1),
        ([2, 3], -2),
        ([2, 3, 4], 0),
        ([2, 3, 4], 2),
        ([2, 3, 4], -3),
        ([0, 0, 0], 0),
        ([0, 0, 0], 1),
        ([1, 32, 64], 1),
        ([1, 1024, 32], 0),
        ([1, 1024, 32], 1),
        ([260, 1, 1], 0),
        ([1024, 1, 32], 0),
        ([1, 1024, 32], 2),
        ([64, 1, 32], 1),
        ([64, 64, 1], 1),
        ([1, 32, 129], 1),
        ([33, 35, 37], 1),
        ([2, 3, 33, 33], 0),
        ([2, 3, 33, 33], 1),
        ([7, 13, 129, 33], 1),
        ([7, 13, 129, 33], 0),
        ([4, 6, 128, 128], 0),
        ([2, 3, 5, 33, 128], 0),
        ([2, 3, 5, 33, 128], 1),
        ([2, 3, 5, 33, 128], 2),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
        (torch.float32, ttnn.bfloat16),
        (torch.int32, ttnn.int32),
    ],
)
def test_cumsum(size, dim, dtypes, device):
    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    # Generate integer input on [-2; 2];
    # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
    # which are not handled yet
    torch_input_tensor = torch.randint(-2, 3, size=size, dtype=torch_dtype)
    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.Layout.TILE)

    expected_output_dtype = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    tensor_rank = len(size)
    # For now, int32 version only supports >3-D tensors and `dim` outher than x and y axes
    if not is_supported(tensor_rank, dim, expected_output_dtype):
        return

    output_tensor = ttnn.experimental.cumsum(input_tensor, dim=dim, dtype=ttnn_dtype)

    assert output_tensor.dtype == expected_output_dtype
    assert output_tensor.shape == (size)

    torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    expected_output = torch.cumsum(torch_input_tensor, dim=dim, dtype=torch_dtype)

    assert_with_pcc(expected_output, torch_output)


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([0], 0),
        ([1], 0),
        ([10], 0),
        ([2, 3], 0),
        ([2, 3], 1),
        ([2, 3], -1),
        ([2, 3], -2),
        ([2, 3, 4], 0),
        ([2, 3, 4], 2),
        ([2, 3, 4], -3),
        ([0, 0, 0], 0),
        ([0, 0, 0], 1),
        ([1, 32, 64], 1),
        ([1, 1024, 32], 0),
        ([1, 1024, 32], 1),
        ([260, 1, 1], 0),
        ([1024, 1, 32], 0),
        ([1, 1024, 32], 2),
        ([64, 1, 32], 1),
        ([64, 64, 1], 1),
        ([1, 32, 129], 1),
        ([33, 35, 37], 1),
        ([2, 3, 33, 33], 0),
        ([2, 3, 33, 33], 1),
        ([7, 13, 129, 33], 1),
        ([7, 13, 129, 33], 0),
        ([4, 6, 128, 128], 0),
        ([2, 3, 2, 2], 1),
        ([2, 3, 2, 2], 0),
        ([2, 1, 33], 0),
        ([2, 3, 5, 33, 128], 0),
        ([2, 3, 5, 33, 128], 1),
        ([2, 3, 5, 33, 128], 2),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
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
        return

    preallocated_output_tensor = ttnn.zeros_like(input_tensor, dtype=ttnn_dtype, layout=ttnn.Layout.TILE)

    output_tensor = ttnn.experimental.cumsum(input_tensor, dim=dim, dtype=ttnn_dtype, output=preallocated_output_tensor)
    torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    expected_output = torch.cumsum(torch_input_tensor, dim=dim, dtype=torch_dtype)

    assert output_tensor.dtype == expected_output_dtype
    assert preallocated_output_tensor.dtype == expected_output_dtype

    assert output_tensor.shape == (size)
    assert preallocated_output_tensor.shape == (size)

    assert preallocated_output_tensor == output_tensor

    assert_with_pcc(expected_output, torch_output)
