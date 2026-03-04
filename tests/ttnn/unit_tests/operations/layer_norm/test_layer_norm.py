# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm Integration Test

Validates that the layer_norm operation:
  1. Imports and runs without Python-side errors
  2. Produces an output tensor with the correct shape and dtype
  3. Accepts optional weight and bias tensors (affine parameters)

Note: With stub kernels the numerical output will be garbage (identity passthrough
for Stage 1). Numerical accuracy is tested in the TDD stage tests once kernels
are implemented.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="32x32-minimal"),
        pytest.param((64, 128), id="64x128-multi-tile"),
        pytest.param((32, 256), id="32x256-wide"),
        pytest.param((128, 64), id="128x64-multi-batch"),
    ],
)
def test_layer_norm_shape(device, shape):
    """Verify output tensor has correct shape and dtype (stub kernels)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input)

    assert list(ttnn_output.shape) == list(shape), f"Output shape {list(ttnn_output.shape)} != expected {list(shape)}"
    assert ttnn_output.dtype == ttnn.bfloat16


def test_layer_norm_with_weight_bias(device):
    """Verify layer_norm accepts weight and bias without crashing."""
    shape = (32, 32)
    W = shape[-1]
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(W, dtype=torch.bfloat16)
    torch_bias = torch.randn(W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_weight = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    ttnn_bias = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    ttnn_output = layer_norm(ttnn_input, weight=ttnn_weight, bias=ttnn_bias)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16


def test_layer_norm_validation_errors(device):
    """Verify that invalid inputs raise descriptive errors."""
    # 1D input (must be 2D)
    with pytest.raises(ValueError, match="2D"):
        x = ttnn.from_torch(
            torch.randn(32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        layer_norm(x)

    # N not multiple of 32
    with pytest.raises(ValueError, match="multiple of 32"):
        x = ttnn.from_torch(
            torch.randn(16, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        layer_norm(x)
