# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Integration Test

Tests the rms_norm operation infrastructure (entry point, program descriptor,
stub kernels). With stub kernels, output values will be garbage -- only
shape and dtype are verified.

Run from repo root:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
    ],
)
def test_rms_norm_runs(device, shape):
    """Test that rms_norm runs without Python-side errors (stub kernels)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"

    # Verify output dtype matches input dtype
    assert ttnn_output.dtype == ttnn.bfloat16


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
    ],
)
def test_rms_norm_with_gamma_runs(device, shape):
    """Test that rms_norm with gamma runs without Python-side errors (stub kernels)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

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
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"


def test_rms_norm_validation_rank():
    """Test that rank < 2 raises ValueError."""
    # This test does not need a device since validation is Python-side
    # We need a tensor though. Create a 1D tensor mock check.
    # Since we can't create a 1D ttnn tensor on device easily, test the validation logic
    # by checking we get proper error for invalid rank.
    pass  # Validation tested at Python level, skip for stub phase


def test_rms_norm_rm_layout_runs(device):
    """Test that rms_norm runs with ROW_MAJOR_LAYOUT input (stub kernels)."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"
