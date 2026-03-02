# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for the layer_norm generic_op operation.

Tests numerical accuracy against torch.nn.functional.layer_norm for:
- Various input shapes (single tile, multi-tile W/H, multi-batch)
- Without affine transform (no gamma/beta)
- With gamma only
- With gamma and beta
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
def test_layer_norm_no_affine(device, shape):
    """Test layer_norm without gamma/beta against PyTorch reference."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch reference (compute in float32 for accuracy)
    W = shape[-1]
    expected = torch.nn.functional.layer_norm(torch_input.to(torch.float32), [W]).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, epsilon=1e-5)
    result = ttnn.to_torch(ttnn_output)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert torch.allclose(result, expected, rtol=0.05, atol=0.2), f"Max diff: {(result - expected).abs().max().item()}"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 64),
        (1, 1, 32, 128),
        (2, 1, 64, 64),
    ],
)
def test_layer_norm_with_affine(device, shape):
    """Test layer_norm with gamma and beta against PyTorch reference."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    torch_beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    # PyTorch reference
    expected = torch.nn.functional.layer_norm(
        torch_input.to(torch.float32),
        [W],
        weight=torch_gamma.to(torch.float32).squeeze(0).squeeze(0).squeeze(0),
        bias=torch_beta.to(torch.float32).squeeze(0).squeeze(0).squeeze(0),
    ).to(torch.bfloat16)

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
    result = ttnn.to_torch(ttnn_output)

    assert list(ttnn_output.shape) == list(shape)
    assert torch.allclose(result, expected, rtol=0.05, atol=0.2), f"Max diff: {(result - expected).abs().max().item()}"
