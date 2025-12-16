# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# Reshape in Tile layout with shapes that are not divisible by 32
@pytest.mark.parametrize(
    "input_shape, output_shape, layout",
    [
        ((1, 15), (15,), ttnn.ROW_MAJOR_LAYOUT),  # RM_last dimension matches, 1D output
        ((2, 1, 1, 1, 15), (2, 15), ttnn.ROW_MAJOR_LAYOUT),  # RM_last dimension matches
        ((16, 1, 1, 247, 13), (1, 16, 247, 13), ttnn.TILE_LAYOUT),  # last two dimensions match
        (
            (16, 1, 1, 256, 16),
            (8, 16, 32, 16),
            ttnn.TILE_LAYOUT,
        ),  # last dimension match but second last multiple of 32 but does not match
        ((32, 32, 32, 15), (32768, 15), ttnn.TILE_LAYOUT),  # Very large tensor
    ],
)
def test_view(input_shape, output_shape, layout, device):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_result = torch_input_tensor.reshape(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn.view(input_tensor, output_shape)
    assert layout == ttnn_output.layout
    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, output_shape, layout",
    [
        ((2, 1, 1, 1, 15), (1, 30), ttnn.ROW_MAJOR_LAYOUT),  # RM last dimension doesn't match
        (
            (16, 1, 256, 1, 16),
            (8, 16, 32, 16),
            ttnn.TILE_LAYOUT,
        ),  # TILE last dimension match but second last does not match, shape mult of 32 only
        (
            (16, 1, 1, 256, 16),
            (8, 16, 32, 1, 16),
            ttnn.TILE_LAYOUT,
        ),  # TILE last dimension match but second last does not match, tensor mult of 32 only
        (
            (256, 1, 1, 16, 16),
            (8, 16, 32, 1, 16),
            ttnn.TILE_LAYOUT,
        ),  # TILE last dimension match but second last does not match, none mult of 32
        (
            (16, 8, 1, 32, 16),
            (8, 16, 31, 16),
            ttnn.TILE_LAYOUT,
        ),  # Volume doesn't match but padded volume does
    ],
)
def test_invalid_cases(input_shape, output_shape, layout, device):
    # Verifies invalid cases do cause an assertion
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=layout, device=device)
    with pytest.raises(RuntimeError):
        ttnn.view(input_tensor, output_shape)
