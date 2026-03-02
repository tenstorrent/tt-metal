# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for the layer_norm generic_op operation.

Tests:
- Basic shape correctness with stub kernels
- Operation with and without gamma/beta
- Shape preservation across various input sizes

Note: With stub kernels, numerical output will be garbage.
These tests verify the Python infrastructure (shape, dtype, no crashes).
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 32, 64),
        (1, 1, 32, 128),
        (1, 1, 128, 64),
        (2, 1, 64, 64),
    ],
)
def test_layer_norm_runs(device, shape):
    """Verify layer_norm executes without Python-side errors and preserves shape."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, epsilon=1e-5)

    # Shape must be preserved
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    # Dtype must be preserved
    assert ttnn_output.dtype == ttnn.bfloat16


def test_layer_norm_with_gamma(device):
    """Verify layer_norm with gamma (scale) tensor runs without errors."""
    shape = (1, 1, 32, 64)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, weight=ttnn_gamma, epsilon=1e-5)

    assert list(ttnn_output.shape) == list(shape)


def test_layer_norm_with_gamma_and_beta(device):
    """Verify layer_norm with gamma and beta tensors runs without errors."""
    shape = (1, 1, 32, 64)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
    torch_beta = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, weight=ttnn_gamma, bias=ttnn_beta, epsilon=1e-5)

    assert list(ttnn_output.shape) == list(shape)
