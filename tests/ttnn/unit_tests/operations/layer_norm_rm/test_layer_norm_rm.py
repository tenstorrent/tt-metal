# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Integration Tests

Validates the Python infrastructure (entry point, program descriptor, kernel setup)
by running with stub kernels. Output values will be garbage since kernels are stubs,
but shape and dtype must be correct, and no Python-side errors should occur.

Run from repo root:
    pytest tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py -v
"""

import pytest
import torch
import ttnn

from .layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
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

    # Verify output shape matches input
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
    ],
)
def test_layer_norm_rm_with_gamma_beta(device, shape):
    """Test that layer_norm_rm with gamma and beta executes without errors."""
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

    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"


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

    with pytest.raises(ValueError, match="row-major"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validation_gamma_width(device):
    """Test that gamma width mismatch raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    # Gamma has width 32, but input has width 64
    torch_gamma = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)

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
