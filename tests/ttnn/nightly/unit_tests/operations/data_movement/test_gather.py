# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_negative_dim(device, shape, dim, dtype):
    """
    Regression test for negative dimension support in ttnn.gather.
    Verifies that negative dimension values work correctly and match PyTorch behavior.
    """
    torch_input = torch.rand(shape, dtype=dtype)

    # Create index tensor for gathering
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)  # Gather subset
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)

    # PyTorch reference with negative dim
    torch_output_neg = torch.gather(torch_input, dim, torch_index.long())

    # Convert to ttnn using the matching test dtype
    torch_to_ttnn_dtype = {
        torch.bfloat16: ttnn.bfloat16,
        torch.float32: ttnn.float32,
    }
    ttnn_dtype = torch_to_ttnn_dtype[dtype]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    # Test with negative dim
    ttnn_output = ttnn.gather(ttnn_input, dim, ttnn_index)
    output = ttnn.to_torch(ttnn_output)

    assert (
        output.shape == torch_output_neg.shape
    ), f"Output shape {output.shape} does not match expected {torch_output_neg.shape}"
    # Use tighter tolerance for float32
    pcc_threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert_with_pcc(torch_output_neg, output, pcc_threshold)


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((4, 128), -1),  # Last dimension
        ((4, 128), -2),  # First dimension
        ((2, 3, 4), -1),  # 3D tensor, last dim
        ((2, 3, 4), -2),  # 3D tensor, middle dim
        ((2, 3, 4), -3),  # 3D tensor, first dim
    ],
)
def test_gather_negative_dim_equals_positive(device, shape, dim):
    """
    Verify that negative and positive dim produce identical results.
    """
    positive_dim = len(shape) + dim

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)

    # Get results for both negative and positive dims
    torch_output_neg = torch.gather(torch_input, dim, torch_index.long())
    torch_output_pos = torch.gather(torch_input, positive_dim, torch_index.long())

    # Verify PyTorch results match
    assert torch.allclose(torch_output_neg, torch_output_pos)

    # Test ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    ttnn_output_neg = ttnn.to_torch(ttnn.gather(ttnn_input, dim, ttnn_index))
    ttnn_output_pos = ttnn.to_torch(ttnn.gather(ttnn_input, positive_dim, ttnn_index))

    assert_with_pcc(ttnn_output_neg, ttnn_output_pos, 0.9999)
    assert_with_pcc(torch_output_neg, ttnn_output_neg, 0.999)
