# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Stage 1: data_pipeline

Reader/writer data pipeline with tilize/untilize. Identity passthrough: output equals input.

Run:
    scripts/tt-test.sh --dev tests/ttnn/unit_tests/operations/rms_norm/test_stage_data_pipeline.py
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
def test_stage_data_pipeline(device, shape, layout):
    """
    Stage 1: Identity passthrough.
    Verifies the data pipeline (reader -> compute -> writer) with tilize/untilize.
    Output should equal input (identity).

    Reference: golden = input_tensor.clone()
    Tolerances: rtol=0.01, atol=0.01
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

    # Verify output shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Compare with reference (identity)
    golden = torch_input.clone()
    torch_output = ttnn.to_torch(ttnn_output).reshape(shape)

    assert torch.allclose(
        torch_output.float(),
        golden.float(),
        rtol=0.01,
        atol=0.01,
    ), f"Stage data_pipeline failed. Max diff: {(torch_output.float() - golden.float()).abs().max()}"
