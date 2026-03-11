# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax Operation - Integration Tests

Tests the softmax operation entry point with stub kernels.
With stub kernels, output values will be garbage -- only shape/dtype/no-crash
are verified. Numerical tests become meaningful once kernels are implemented.
"""

import pytest
import torch
import ttnn

from ttnn.operations.softmax import softmax


# ==============================================================================
# Shape and execution tests (work with stub kernels)
# ==============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 32, 128), id="multi_tile_W"),
        pytest.param((1, 1, 128, 32), id="multi_tile_H"),
        pytest.param((1, 1, 64, 256), id="non_square"),
        pytest.param((2, 3, 64, 64), id="multi_batch"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_w", "dim_h"])
def test_softmax_runs(device, shape, dim):
    """Verify softmax runs without errors and output shape is correct."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=dim)

    # Verify output shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"

    # Verify output dtype
    assert ttnn_output.dtype == ttnn.bfloat16


@pytest.mark.parametrize("numeric_stable", [True, False], ids=["stable", "unstable"])
def test_softmax_numeric_stable_flag(device, numeric_stable):
    """Verify softmax accepts numeric_stable parameter without errors."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable)
    assert list(ttnn_output.shape) == list(shape)


# ==============================================================================
# Input validation tests (these do NOT require device)
# ==============================================================================


def test_softmax_invalid_dim(device):
    """Verify ValueError for invalid dim."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="dim must be -1 or -2"):
        softmax(ttnn_input, dim=0)


def test_softmax_invalid_rank(device):
    """Verify ValueError for non-4D input."""
    shape = (32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="must be 4D"):
        softmax(ttnn_input)


def test_softmax_invalid_layout(device):
    """Verify ValueError for non-TILE_LAYOUT input."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    # Create ROW_MAJOR tensor on device
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="must be TILE_LAYOUT"):
        softmax(ttnn_input)
