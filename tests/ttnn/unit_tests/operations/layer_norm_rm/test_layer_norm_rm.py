# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Integration Test

Tests the full layer_norm_rm operation with stub kernels.
With stubs, output content is undefined (garbage), but the operation
must execute without Python-side errors and preserve shape/dtype.
"""

import pytest
import torch
import ttnn

from .layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Verify layer_norm_rm executes without errors and preserves shape."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output shape matches input shape
    expected_shape = list(shape)
    assert (
        list(ttnn_output.shape) == expected_shape
    ), f"Shape mismatch: {list(ttnn_output.shape)} vs expected {expected_shape}"

    # Verify output dtype
    assert ttnn_output.dtype == ttnn.bfloat16, f"Dtype mismatch: {ttnn_output.dtype} vs expected bfloat16"

    # Verify output layout
    assert (
        ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: {ttnn_output.layout} vs expected ROW_MAJOR_LAYOUT"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_with_affine"),
    ],
)
def test_layer_norm_rm_with_gamma_beta_runs(device, shape):
    """Verify layer_norm_rm with gamma/beta executes without errors."""
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

    # Verify output shape matches input shape
    expected_shape = list(shape)
    assert (
        list(ttnn_output.shape) == expected_shape
    ), f"Shape mismatch: {list(ttnn_output.shape)} vs expected {expected_shape}"


def test_layer_norm_rm_validation_dtype(device):
    """Verify validation rejects non-bfloat16 input."""
    torch_input = torch.randn(1, 1, 32, 32)
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
    """Verify validation rejects non-row-major input."""
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
