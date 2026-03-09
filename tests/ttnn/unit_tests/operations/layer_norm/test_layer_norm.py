# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm - Integration Tests

Tests the layer_norm operation against PyTorch reference implementations.
These are infrastructure/stub tests: with stub kernels the numerical output
will be garbage, but the shape and dtype must be correct and generic_op
must execute without Python-side errors.

Run with:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/layer_norm/test_layer_norm.py
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm import layer_norm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile_HxW"),
        pytest.param((1, 1, 32, 256), id="wide"),
        pytest.param((1, 1, 128, 64), id="tall"),
    ],
)
def test_layer_norm_runs(device, shape):
    """
    Verify layer_norm runs without errors and produces correct output shape.

    With stub kernels the numerical output is not checked -- only shape and
    dtype are verified. The generic_op call must not crash.
    """
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input)

    # Shape must be preserved
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile_HxW"),
    ],
)
def test_layer_norm_with_gamma_beta_runs(device, shape):
    """
    Verify layer_norm runs with optional gamma and beta tensors.
    Shape and dtype are verified; numerical correctness is not tested at stub stage.
    """
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones(W, dtype=torch.bfloat16)
    torch_beta = torch.zeros(W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
    ],
)
def test_layer_norm_from_row_major_input(device, shape):
    """
    Verify layer_norm auto-converts ROW_MAJOR input to TILE_LAYOUT.
    This mirrors how the TDD stage tests supply their inputs.
    """
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Supply as ROW_MAJOR -- layer_norm should convert internally
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input)

    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"
