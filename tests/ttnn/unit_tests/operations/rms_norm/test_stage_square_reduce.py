# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Stage 2: square_reduce

Add square and reduce_row phases. Output is mean(x^2) per row with reduced output shape.

Run:
    scripts/tt-test.sh --dev tests/ttnn/unit_tests/operations/rms_norm/test_stage_square_reduce.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, id="tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
    ],
)
def test_stage_square_reduce(device, shape, layout):
    """
    Stage 2: Square + reduce_row.
    Output is mean(x^2, dim=-1, keepdim=True) with tile-aligned reduced shape.

    Reference: golden = (input_tensor ** 2).mean(dim=-1, keepdim=True)
    Output shape: list(shape[:-1]) + [32]
    Compare slice: [:,:,:,0:1]
    Tolerances: rtol=0.02, atol=0.1
    """
    # No gamma for this stage
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input)

    # Expected output shape: tile-aligned reduced
    expected_shape = list(shape[:-1]) + [32]
    assert list(ttnn_output.shape) == expected_shape, f"Shape mismatch: {list(ttnn_output.shape)} vs {expected_shape}"

    # Compare with reference using compare slice [:,:,:,0:1]
    golden = (torch_input.float() ** 2).mean(dim=-1, keepdim=True)
    torch_output = ttnn.to_torch(ttnn_output).reshape(expected_shape)

    # Extract column 0 for comparison
    output_slice = torch_output[:, :, :, 0:1].float()
    golden_slice = golden.float()

    assert torch.allclose(
        output_slice,
        golden_slice,
        rtol=0.02,
        atol=0.1,
    ), f"Stage square_reduce failed. Max diff: {(output_slice - golden_slice).abs().max()}"
