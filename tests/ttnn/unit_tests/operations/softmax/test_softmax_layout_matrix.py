# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layout test matrix for softmax.

Tests TILE_LAYOUT and ROW_MAJOR_LAYOUT support across shapes, dtypes,
and dims. This is the single authoritative layout-correctness test for
softmax, per the /memory-layouts skill §7.

Block formats (bfloat8_b) are skipped for ROW_MAJOR — they have no RM
representation (structural impossibility, also in feature_spec.INVALID).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch reference using the numerically-stable form."""
    return torch.softmax(input_tensor.float(), dim=dim)


# PCC thresholds keyed by dtype — same as the golden suite.
PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="32x32_small"),
        pytest.param((1, 1, 32, 64), id="32x64_aligned"),
        pytest.param((1, 1, 64, 128), id="64x128_aligned"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
        pytest.param((2, 3, 96, 96), id="multi_batch_square"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, id="tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_layout_matrix(device, shape, layout, dtype, dim):
    """Test softmax across layout × shape × dtype × dim.

    ROW_MAJOR path: reader reads sticks → compute tilizes → softmax math →
    compute untilizes → writer writes sticks. The math is identical to
    the TILE path; only the data-access boundary changes.
    """
    # Block formats do not support ROW_MAJOR layout
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        pytest.skip("Block formats do not support ROW_MAJOR layout")

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    # Output layout must match input layout
    assert ttnn_output.layout == layout, f"Layout mismatch: {ttnn_output.layout} != {layout}"
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"
    assert ttnn_output.dtype == dtype, f"dtype mismatch: {ttnn_output.dtype} != {dtype}"

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[dtype])


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, id="tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
    ],
)
def test_softmax_layout_l1_memory(device, shape, layout, dtype):
    """Test softmax with L1 memory config across layouts."""
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        pytest.skip("Block formats do not support ROW_MAJOR layout")

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert ttnn_output.layout == layout, f"Layout mismatch: {ttnn_output.layout} != {layout}"

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[dtype])
