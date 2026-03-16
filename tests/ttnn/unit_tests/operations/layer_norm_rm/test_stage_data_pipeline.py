# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 1: data_pipeline
Tilize RM input to tiles, then untilize back to RM (identity passthrough).

Reference: return input_tensor
Tolerances: rtol=0.01, atol=0.01
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


def pytorch_reference(input_tensor):
    """Stage 1 reference: identity passthrough."""
    return input_tensor


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_1x1x32x32"),
        pytest.param((1, 1, 64, 128), id="multi_tile_1x1x64x128"),
        pytest.param((1, 1, 32, 256), id="non_square_1x1x32x256"),
        pytest.param((4, 2, 64, 64), id="multi_batch_4x2x64x64"),
    ],
)
def test_stage_data_pipeline(device, shape):
    """Test tilize+untilize identity passthrough."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Compare with reference
    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = pytorch_reference(torch_input)

    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=0.01,
        atol=0.01,
    ), f"Output mismatch. Max diff: {(torch_output.float() - torch_expected.float()).abs().max()}"
