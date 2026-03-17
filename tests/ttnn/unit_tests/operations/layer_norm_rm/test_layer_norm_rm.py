# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Integration Test

Tests the full layer_norm_rm operation infrastructure.
With stub kernels, output values will be garbage -- we only verify:
1. The operation runs without Python-side errors
2. Output tensor has the correct shape and dtype
3. Output tensor is ROW_MAJOR_LAYOUT

Run from repo root:
    pytest tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py -v
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
        pytest.param((1, 1, 32, 512), id="wide"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Test that layer_norm_rm executes without Python-side errors (stub kernels)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output shape and properties
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
def test_layer_norm_rm_with_gamma_beta(device, shape):
    """Test that layer_norm_rm with gamma/beta executes without Python-side errors."""
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    torch_beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=1e-5)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


def test_layer_norm_rm_validation_dtype(device):
    """Test that non-bfloat16 input raises ValueError."""
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


def test_layer_norm_rm_validation_layout(device):
    """Test that TILE_LAYOUT input raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="ROW_MAJOR_LAYOUT"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validation_gamma_width(device):
    """Test that gamma width mismatch raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)  # wrong width

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="width"):
        layer_norm_rm(ttnn_input, ttnn_gamma)
