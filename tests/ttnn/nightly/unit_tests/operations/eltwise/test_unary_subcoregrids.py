# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            (torch.Size([1, 2, 32, 960])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 7, 32, 96])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 8, 32, 128])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 17, 32, 32])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 1, 32, 32 * 1024])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op, atol_threshold, high, low",
    [
        (ttnn.log, 4e-2, 100, 1),
        (ttnn.log10, 4e-2, 100, 1),
        (ttnn.log1p, 4e-2, 100, 1),
        (ttnn.log2, 7e-2, 100, 1),
    ],
)
def test_unary_subcore_grid(device, shape, sub_core_grid, dtype, ttnn_op, atol_threshold, high, low):
    """Test unary operations with sub_core_grids parameter"""
    torch.manual_seed(0)
    torch_dtype = dtype
    ttnn_dtype = ttnn.bfloat16

    num_elements = torch.prod(torch.tensor(shape)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch_dtype)
    torch_input = torch_input[:num_elements].reshape(shape)

    # Get golden result from torch
    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation with sub_core_grids
    ttnn_output = ttnn_op(ttnn_input, sub_core_grids=sub_core_grid)

    # Convert output back to torch
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Compare with golden using specified thresholds
    all_close_assert = torch.allclose(ttnn_output, torch_output, atol=atol_threshold)
    assert all_close_assert


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            (torch.Size([1, 2, 32, 960])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 7, 32, 96])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 8, 32, 128])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 17, 32, 32])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 1, 32, 32 * 1024])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        # (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.floor,
        ttnn.trunc,
        ttnn.ltz,
    ],
)
def test_unary_subcore_grids(device, shape, sub_core_grid, torch_dtype, ttnn_dtype, ttnn_op):
    torch.manual_seed(0)
    torch_input = torch.empty(shape, dtype=torch_dtype).uniform_(-100, 100)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_output = ttnn_op(ttnn_input, sub_core_grids=sub_core_grid)
    output_tensor = ttnn.to_torch(ttnn_output)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(torch_input)
    assert_with_pcc(output_tensor, golden_tensor)
