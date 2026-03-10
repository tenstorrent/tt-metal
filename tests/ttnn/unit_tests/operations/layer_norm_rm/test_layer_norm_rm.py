# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Integration Test

Tests the layer_norm_rm operation end-to-end. With stub kernels,
numerical output will be garbage, but the test verifies:
  - Import and invocation work
  - Program descriptor builds without errors
  - ttnn.generic_op() executes without Python-side crashes
  - Output tensor has correct shape and dtype
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 32, 128), id="multi_tile_w"),
        pytest.param((1, 1, 64, 128), id="multi_row"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Verify that layer_norm_rm executes without Python-side errors (stub kernels)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify shape preserved
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: {list(ttnn_output.shape)} vs expected {list(shape)}"

    # Verify dtype preserved
    assert ttnn_output.dtype == ttnn.bfloat16


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
    ],
)
def test_layer_norm_rm_with_gamma_beta(device, shape):
    """Verify layer_norm_rm with gamma/beta executes without Python-side errors."""
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

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16


def test_layer_norm_rm_validation_dtype(device):
    """Verify that non-bfloat16 input raises ValueError."""
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
    """Verify that tile-layout input raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="row-major"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validation_gamma_width(device):
    """Verify that gamma width mismatch raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)  # Wrong width

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

    with pytest.raises(ValueError, match="gamma width"):
        layer_norm_rm(ttnn_input, ttnn_gamma)
