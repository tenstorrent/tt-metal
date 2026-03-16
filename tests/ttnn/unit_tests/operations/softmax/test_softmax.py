# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Integration test for softmax generic_op operation

import pytest
import torch
import ttnn

from ttnn.operations.softmax import softmax


def test_softmax_runs(device):
    """Verify the softmax operation executes without errors and produces correct output shape."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation (with stub kernels, output will be garbage)
    ttnn_output = softmax(ttnn_input)

    # Verify output shape is correct
    assert list(ttnn_output.shape) == [1, 1, 32, 32]
    # Verify output dtype
    assert ttnn_output.dtype == ttnn.bfloat16


def test_softmax_multi_tile(device):
    """Verify multi-tile execution."""
    torch_input = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input)
    assert list(ttnn_output.shape) == [1, 1, 64, 128]


def test_softmax_dim_h(device):
    """Verify dim=-2 execution."""
    torch_input = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-2)
    assert list(ttnn_output.shape) == [1, 1, 64, 64]


def test_softmax_validation_dtype(device):
    """Verify dtype validation."""
    torch_input = torch.randn(1, 1, 32, 32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="Input must be bfloat16"):
        softmax(ttnn_input)


def test_softmax_validation_dim(device):
    """Verify dim validation."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="Only dim=-1.*and dim=-2.*supported"):
        softmax(ttnn_input, dim=0)
