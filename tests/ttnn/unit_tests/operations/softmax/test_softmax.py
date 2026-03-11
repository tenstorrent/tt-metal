# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax - Integration Test

Validates that the softmax operation infrastructure (entry point, program descriptor,
kernel stubs) works end-to-end without Python-side errors.

With stub kernels, output values will be garbage -- only shape/dtype are validated.
Numerical correctness is validated in TDD stage tests after kernels are implemented.
"""

import pytest
import torch
import ttnn

from ttnn.operations.softmax import softmax


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 32, 128), id="multi_tile_w"),
        pytest.param((1, 1, 128, 32), id="multi_tile_h"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_softmax_dim_w_runs(device, shape):
    """Test softmax dim=-1 executes without Python-side errors."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-1)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 128, 64), id="multi_tile_h"),
    ],
)
def test_softmax_dim_h_runs(device, shape):
    """Test softmax dim=-2 executes without Python-side errors."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-2)

    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"


def test_softmax_validation_wrong_dtype(device):
    """Test that wrong dtype raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="bfloat16"):
        softmax(ttnn_input)


def test_softmax_validation_wrong_dim(device):
    """Test that invalid dim raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="dim"):
        softmax(ttnn_input, dim=0)
