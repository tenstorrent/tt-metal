# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Integration Test

Tests the full layer_norm_rm operation infrastructure.
With stub kernels, output values will be garbage but shape/dtype must be correct.
Once kernels are implemented, numerical accuracy tests apply.
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
        pytest.param((1, 1, 32, 256), id="wide"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Verify the operation executes without Python-side errors and output shape is correct."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output shape and dtype
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
    ],
)
def test_layer_norm_rm_with_gamma_beta(device, shape):
    """Verify the operation runs with gamma and beta tensors."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_ttnn = ttnn.from_torch(
        gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    beta_ttnn = ttnn.from_torch(
        beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    ttnn_output = layer_norm_rm(ttnn_input, gamma=gamma_ttnn, beta=beta_ttnn)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16


def test_layer_norm_rm_validation_dtype(device):
    """Verify ValueError when input is not bfloat16."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with pytest.raises(ValueError, match="bfloat16"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validation_layout(device):
    """Verify ValueError when input is tile layout."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    with pytest.raises(ValueError, match="row-major"):
        layer_norm_rm(ttnn_input)


def test_layer_norm_rm_validation_gamma_width(device):
    """Verify ValueError when gamma width does not match input width."""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    gamma_wrong = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    gamma_ttnn = ttnn.from_torch(
        gamma_wrong,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with pytest.raises(ValueError, match="gamma width"):
        layer_norm_rm(ttnn_input, gamma=gamma_ttnn)
