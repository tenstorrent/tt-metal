# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 4: variance_inv_std

Compute adds square, reduce_var, add_eps+rsqrt, and mul_inv_std phases.
Produces full layer normalization output without affine transform.

Reference: torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=None, bias=None, eps=1e-5)
Tolerances: rtol=0.05, atol=0.2
"""

import pytest
import torch
import torch.nn.functional
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
        pytest.param((1, 1, 64, 128), id="multi_tile_hw"),
        pytest.param((1, 1, 32, 256), id="wide"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_stage_variance_inv_std(device, shape):
    """Stage 4: Full layer normalization without affine."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Reference: layer norm without affine
    expected = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]], weight=None, bias=None, eps=1e-5)
    actual = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        actual.float(), expected.float(), rtol=0.05, atol=0.2
    ), f"Stage 4 variance_inv_std failed. Max diff: {(actual.float() - expected.float()).abs().max()}"
