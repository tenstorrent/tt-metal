# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 4: variance_rsqrt

Add square, reduce_row for variance, add_eps+rsqrt, multiply by rstd.
Full layer norm without affine transform.

Reference: torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=1e-5)
Tolerance: rtol=0.05, atol=0.2
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="1_1_32_32"),
        pytest.param((1, 1, 64, 128), id="1_1_64_128"),
        pytest.param((1, 1, 32, 256), id="1_1_32_256"),
        pytest.param((4, 2, 64, 64), id="4_2_64_64"),
        pytest.param((1, 1, 32, 512), id="1_1_32_512"),
    ],
)
def test_stage_variance_rsqrt(device, shape):
    """Stage 4: Full layer norm without affine."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, epsilon=1e-5)

    # Verify shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Reference: layer_norm without weight/bias
    x = torch_input.float()
    expected = F.layer_norm(x, [x.shape[-1]], eps=1e-5)

    torch_output = ttnn.to_torch(ttnn_output).float()

    assert torch.allclose(torch_output, expected, rtol=0.05, atol=0.2), (
        f"Stage variance_rsqrt FAILED for shape {shape}. "
        f"Max diff: {(torch_output - expected).abs().max().item():.6f}"
    )
