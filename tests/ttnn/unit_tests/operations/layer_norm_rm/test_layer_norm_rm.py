# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Integration Test

Validates that the operation:
  1. Can be imported without errors
  2. Creates a program descriptor without errors
  3. Invokes ttnn.generic_op without Python-side crashes
  4. Produces an output tensor with the correct shape

Note: With stub kernels, the output values will be uninitialized garbage.
Shape and dtype verification are the meaningful checks at this stage.

Run with:
    ./tt-test.sh tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 32, 64), id="multi_tile_W"),
        pytest.param((1, 1, 64, 128), id="multi_tile_HW"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((2, 1, 64, 64), id="multi_batch"),
    ],
)
def test_layer_norm_rm_shape(device, shape):
    """Verify operation runs and output shape matches input shape."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run without gamma/beta (stub kernels produce garbage values but should not crash)
    ttnn_output = layer_norm_rm(ttnn_input)

    expected_shape = list(shape)
    assert (
        list(ttnn_output.shape) == expected_shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {expected_shape}"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_with_affine"),
        pytest.param((1, 1, 64, 128), id="multi_tile_with_affine"),
    ],
)
def test_layer_norm_rm_with_gamma_beta_shape(device, shape):
    """Verify operation runs with gamma and beta tensors and output shape is correct."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.ones(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)

    expected_shape = list(shape)
    assert (
        list(ttnn_output.shape) == expected_shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {expected_shape}"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


def test_layer_norm_rm_validation_errors(device):
    """Verify input validation raises appropriate errors."""
    # Test: wrong layout should raise ValueError
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input_tile = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="row-major"):
        layer_norm_rm(ttnn_input_tile)

    # Test: non-tile-aligned width should raise ValueError
    # (width = 33 is not multiple of 32, but must create 33-wide tensor)
    # This is hard to test without a valid tensor. Skip this edge case for stub validation.
