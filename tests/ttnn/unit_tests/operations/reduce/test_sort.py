# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE_WIDTH = 32


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        # ([64, 64], -1, False),
        # ([32, 128], -1, False),
        # ([1, 1, 32, 64], -1, True),
        # ([32, 128], 1, True),
        # ([1], 0, True),
        # ([], -1, True),
        # ([1, 1, 32, 64], -1, False),
        # ([1, 2048, 1, 64], -1, False),
        # ([1, 55, 43], -1, True),
        # ([11, 29, 14, 1], -1, True),
        # ([1, 1, 512, 64], -1, False),
        # ([1, 1, 2112, 64], -1, False),
    ],
)
def test_sort_standard(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_with_pcc(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        # ([64, 64], -1, False),
        # ([32, 128], -1, False),
        # ([1, 1, 32, 64], -1, True),
        # ([32, 128], 1, True),
        # ([1], 0, True),
        # ([], -1, True),
        # ([1, 1, 32, 64], -1, False),
        # ([1, 2048, 1, 64], -1, False),
        # ([1, 55, 43], -1, True),
        # ([11, 29, 14, 1], -1, True),
        # ([1, 1, 512, 64], -1, False),
        # ([1, 1, 2112, 64], -1, False),
    ],
)
def test_sort_prealocated_output(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    ttnn_sort_values = ttnn.zeros_like(ttnn_input)
    ttnn_sort_indices = ttnn.zeros_like(ttnn_input)
    ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending, out=(ttnn_sort_values, ttnn_sort_indices))

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_with_pcc(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


def test_sort_matrix_3x3(device):
    torch_dtype = torch.bfloat16
    
    # input = torch.tensor([[9, 3, 5], [4, 7, 1], [8, 2, 6]], dtype=torch_dtype)
    # input = torch.transpose(torch.arange(1, 64 * 64 + 1, dtype=torch_dtype).reshape(64, 64), 0, 1)
    # input = torch.randint(0, 100, [32, 4 * TILE_WIDTH], dtype=torch_dtype)
    input = torch.arange(128, dtype=torch_dtype).repeat(32, 1)
    # input = torch.randn([1, 2, 3, 4, 1, 1, 1], dtype=torch_dtype)
    
    print(f"1. Input shape: {input.shape}")
    print(f"2. Tensor rank: {input.ndim}")
    
    torch.set_printoptions(profile="full")
    # print(f"3. Input tensor row:  {input[0]}")
    torch.set_printoptions(profile="default")

    
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=-1, descending=False)
    
    torch.set_printoptions(profile="full")
    print(f"4. Torch sorted values: {torch_sort_values[0]}")
    # print(f"5. Torch sorted indices: {torch_sort_indices[0]}")
    torch.set_printoptions(profile="default")  # Reset to default after printing

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=-1)
    # ttnn_sort_values, ttnn_sort_indices = ttnn.topk(ttnn_input, k=64, dim=-1)

    # print(f"    > 6. ttnn_sort_values: {ttnn_sort_values}")
    # print(f"    > 7. ttnn_sort_indices: {ttnn_sort_indices}")
    
    out = ttnn.from_device(ttnn_sort_values).to_torch()
    out_idx = ttnn.from_device(ttnn_sort_indices).to_torch()
    
    torch.set_printoptions(profile="full")
    print(f"8. TTNN sorted values: {out[0]}")
    # print(f"9. TTNN sorted indices: {out_idx[0]}")
    torch.set_printoptions(profile="default")  # Reset to default after printing