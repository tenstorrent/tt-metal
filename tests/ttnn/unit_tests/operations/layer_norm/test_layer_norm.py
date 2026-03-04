# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for layer_norm.

These tests validate:
1. Output shape and dtype correctness.
2. Numerical accuracy vs torch.nn.functional.layer_norm (once kernels are implemented).

Note: with stub kernels the numerical comparison will fail; only shape/dtype
checks are expected to pass at this stage.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="1x1x32x32"),
        pytest.param((1, 1, 64, 128), id="1x1x64x128"),
        pytest.param((1, 1, 32, 256), id="1x1x32x256"),
    ],
)
def test_layer_norm_shape_no_affine(device, shape):
    """Verify output shape with no gamma/beta (no-affine mode)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input)

    # Shape must match
    assert list(ttnn_output.shape) == list(shape), f"Expected shape {list(shape)}, got {list(ttnn_output.shape)}"
    # Dtype must match
    assert ttnn_output.dtype == ttnn.bfloat16


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="1x1x32x32"),
        pytest.param((1, 1, 64, 128), id="1x1x64x128"),
    ],
)
def test_layer_norm_shape_with_affine(device, shape):
    """Verify output shape with gamma and beta."""
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
    beta_torch = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tt = ttnn.from_torch(
        gamma_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    beta_tt = ttnn.from_torch(
        beta_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn_output = layer_norm(ttnn_input, gamma=gamma_tt, beta=beta_tt)

    assert list(ttnn_output.shape) == list(shape), f"Expected shape {list(shape)}, got {list(ttnn_output.shape)}"
    assert ttnn_output.dtype == ttnn.bfloat16
