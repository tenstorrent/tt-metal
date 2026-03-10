# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for layer_norm_rm.

Verifies:
- Operation runs without Python-side errors
- Output tensor has correct shape and dtype
- Operation accepts optional gamma/beta

NOTE: With stub kernels, output values will be garbage (no math executed).
Shape/dtype checks are the meaningful verification at this stage.

Run with:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_layer_norm_rm_shape(device, shape):
    """Verify output shape matches input shape."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"
    assert ttnn_output.dtype == ttnn.bfloat16, f"dtype mismatch: got {ttnn_output.dtype}, expected bfloat16"


def test_layer_norm_rm_runs_minimal(device):
    """Minimal smoke test: ensure op executes without Python-side errors."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    assert list(ttnn_output.shape) == list(shape)


def test_layer_norm_rm_with_gamma_beta(device):
    """Verify op accepts gamma and beta without errors."""
    shape = (1, 1, 32, 32)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma_1d = torch.ones(W, dtype=torch.bfloat16)
    beta_1d = torch.zeros(W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        gamma_1d.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        beta_1d.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)

    assert list(ttnn_output.shape) == list(shape)


def test_layer_norm_rm_validation_wrong_dtype(device):
    """Verify ValueError for non-bfloat16 input."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="bfloat16"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validation_tile_layout(device):
    """Verify ValueError for TILE_LAYOUT input."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="ROW_MAJOR"):
        layer_norm_rm(ttnn_input)
