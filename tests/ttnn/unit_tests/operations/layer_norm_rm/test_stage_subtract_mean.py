# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 3: subtract_mean

Add sub(input, mean) with COL broadcast.
Output is centered values (input - row_mean).

Reference: x - x.mean(dim=-1, keepdim=True)
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
def test_stage_subtract_mean(device, shape):
    """Stage 3: Output = input - row_mean (centered values)."""
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

    # Reference: centered values
    x = torch_input.float()
    expected = x - x.mean(dim=-1, keepdim=True)

    torch_output = ttnn.to_torch(ttnn_output).float()

    assert torch.allclose(torch_output, expected, rtol=0.02, atol=0.1), (
        f"Stage subtract_mean FAILED for shape {shape}. "
        f"Max diff: {(torch_output - expected).abs().max().item():.6f}"
    )
