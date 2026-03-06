# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Integration Tests

Run with:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py

Tests live in the tests/ directory, separate from operation code.
Uses the `device` fixture from conftest for device management.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_1x1x32x32"),
        pytest.param((1, 1, 64, 128), id="multi_tile_1x1x64x128"),
        pytest.param((1, 1, 32, 256), id="wide_1x1x32x256"),
        pytest.param((4, 2, 64, 64), id="batch_4x2x64x64"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """
    Verify that layer_norm_rm executes without errors and produces
    output with the correct shape. With stub kernels, output values
    are garbage - shape verification only.
    """
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run with stub kernels (output will be uninitialized memory, not numerically correct)
    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: {list(ttnn_output.shape)} vs expected {list(shape)}"

    # Verify output dtype
    assert ttnn_output.dtype == ttnn.bfloat16, f"Dtype mismatch: {ttnn_output.dtype} vs expected bfloat16"

    # Verify output layout
    assert (
        ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: {ttnn_output.layout} vs expected ROW_MAJOR_LAYOUT"


def test_layer_norm_rm_shape_minimal(device):
    """Minimal single-tile test: verify operation runs and output shape is preserved."""
    shape = (1, 1, 32, 32)
    torch_input = torch.ones(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)
    assert list(ttnn_output.shape) == list(shape)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
def test_layer_norm_rm_with_gamma_beta_runs(device, shape):
    """
    Verify that layer_norm_rm with gamma and beta executes without errors.
    Output shape must match input shape.
    """
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
    torch_beta = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)

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

    ttnn_output = layer_norm_rm(ttnn_input, gamma=ttnn_gamma, beta=ttnn_beta)

    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: {list(ttnn_output.shape)} vs expected {list(shape)}"
