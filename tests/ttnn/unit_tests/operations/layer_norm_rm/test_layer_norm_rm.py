# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Integration Test

Tests the full layer_norm_rm entry point with stub kernels.
With stubs the numerical output is garbage, but we verify:
  - No Python-side errors (import, allocation, program descriptor creation)
  - generic_op executes without crashing
  - Output tensor has correct shape, dtype, layout
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_32x32"),
        pytest.param((1, 1, 64, 128), id="multi_tile_64x128"),
        pytest.param((1, 1, 32, 256), id="wide_32x256"),
        pytest.param((4, 2, 64, 64), id="batch_4x2x64x64"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Verify layer_norm_rm executes and returns correct shape (stub kernels)."""
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

    # Verify shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"

    # Verify dtype
    assert ttnn_output.dtype == ttnn.bfloat16

    # Verify layout
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 64), id="with_affine_32x64"),
    ],
)
def test_layer_norm_rm_with_gamma_beta(device, shape):
    """Verify layer_norm_rm with gamma/beta executes (stub kernels)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    W = shape[-1]
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tt = ttnn.from_torch(
        gamma_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    beta_tt = ttnn.from_torch(
        beta_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    ttnn_output = layer_norm_rm(ttnn_input, gamma=gamma_tt, beta=beta_tt)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


def test_layer_norm_rm_validation_rejects_tile_layout(device):
    """Verify that TILE_LAYOUT input is rejected."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    with pytest.raises(ValueError, match="ROW_MAJOR_LAYOUT"):
        layer_norm_rm(ttnn_input)
