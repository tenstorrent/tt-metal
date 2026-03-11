# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Integration Test

Tests the layer_norm_rm operation infrastructure:
- Entry point imports and validates correctly
- Program descriptor creates without errors
- generic_op executes without Python-side crashes
- Output tensor has correct shape and dtype

Note: With stub kernels, output values will be garbage. Numerical
correctness is verified in TDD stage tests after kernel implementation.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="wide"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Verify operation runs and output shape is correct (stub kernels)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: {list(ttnn_output.shape)} vs expected {list(shape)}"

    # Verify output dtype
    assert ttnn_output.dtype == ttnn.bfloat16

    # Verify output layout
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


def test_layer_norm_rm_validates_dtype(device):
    """Verify ValueError for non-bfloat16 input."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="bfloat16"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validates_layout(device):
    """Verify ValueError for non-ROW_MAJOR input."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="ROW_MAJOR"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validates_gamma_width(device):
    """Verify ValueError when gamma width does not match input width."""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Gamma with wrong width (32 vs input width 64)
    torch_gamma = torch.ones(1, 1, 1, 32, dtype=torch.bfloat16)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="gamma width"):
        layer_norm_rm(ttnn_input, gamma=ttnn_gamma)
