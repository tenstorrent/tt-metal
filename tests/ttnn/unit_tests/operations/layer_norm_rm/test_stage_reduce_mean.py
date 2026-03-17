# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 2: reduce_mean

Add REDUCE_ROW for mean computation in compute kernel.
Output is the row-wise mean broadcast back to full input shape.

Reference: x.mean(dim=-1, keepdim=True).expand_as(x)
Tolerance: rtol=0.02, atol=0.1
"""

import pytest
import torch
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
def test_stage_reduce_mean(device, shape):
    """Stage 2: Output = row-wise mean broadcast to input shape."""
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

    # Reference: mean broadcast to input shape
    x = torch_input.float()
    expected = x.mean(dim=-1, keepdim=True).expand_as(x)

    torch_output = ttnn.to_torch(ttnn_output).float()

    assert torch.allclose(torch_output, expected, rtol=0.02, atol=0.1), (
        f"Stage reduce_mean FAILED for shape {shape}. " f"Max diff: {(torch_output - expected).abs().max().item():.6f}"
    )
