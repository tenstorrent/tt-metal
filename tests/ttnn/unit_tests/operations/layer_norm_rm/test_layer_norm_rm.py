# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for layer_norm_rm.

Tests that the operation infrastructure works correctly:
- Tensor allocation
- Program descriptor creation
- generic_op invocation without Python-side errors
- Output shape is correct

Note: With stub kernels the numerical output will be garbage.
Shape checks are the primary validation at this stage.
"""

import pytest
import torch
import ttnn

from .layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 32, 128), id="multi_tile_W"),
        pytest.param((1, 1, 64, 128), id="multi_tile_HW"),
    ],
)
def test_layer_norm_rm_runs(device, shape):
    """Verify layer_norm_rm runs without Python errors and produces correct output shape."""
    torch.manual_seed(0)
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

    # Execute (stub kernels produce garbage output, shape check is what matters)
    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)

    # Verify output shape matches input shape
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    # Verify output dtype
    assert ttnn_output.dtype == ttnn.bfloat16, f"Dtype mismatch: got {ttnn_output.dtype}, expected bfloat16"

    # Verify output layout
    assert (
        ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: got {ttnn_output.layout}, expected ROW_MAJOR_LAYOUT"
