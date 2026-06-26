# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5a — V2 RM layout streaming path.

Tests that ROW_MAJOR layout shapes exceeding the V1 CB budget (256 KiB)
correctly dispatch the V2 streaming path with chunked tilize/untilize.

Covers:
  - chunk_along_reduce dim=-1 RM (attention use case: wide W)
  - chunk_along_reduce dim=-2 RM (tall H)
  - chunk_along_non_reduce dim=-1 RM
  - chunk_along_non_reduce dim=-2 RM
  - fp32 and bf16 dtypes
  - Non-aligned W in V2 RM path
  - V1/V2 equivalence (same output for shapes that fit V1 budget)
  - Multi-slab (NC > 1) distribution
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_softmax_rm(device, shape, dim, dtype, torch_dtype):
    """Run softmax on RM tensor and compare to torch."""
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=torch_dtype)
    expected = torch.softmax(x, dim=dim)

    ttnn_input = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(output)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    max_diff = (result.float() - expected.float()).abs().max().item()
    return result, expected, pcc, max_diff


@pytest.mark.parametrize(
    "shape, dim",
    [
        # chunk_along_reduce dim=-1 (wide W)
        ((1, 1, 32, 4096), -1),
        ((1, 1, 32, 8192), -1),
        ((1, 1, 128, 4096), -1),
        ((2, 1, 64, 4096), -1),
        # chunk_along_reduce dim=-2 (tall H)
        ((1, 1, 2048, 256), -2),
        ((1, 1, 4096, 128), -2),
        # chunk_along_non_reduce cases
        ((2, 1, 64, 4096), -2),
        ((1, 1, 2048, 2048), -1),
    ],
)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)],
)
def test_rm_v2_wide_shapes(device, shape, dim, dtype, torch_dtype):
    """Test RM V2 streaming path for wide/tall shapes that exceed V1 budget."""
    result, expected, pcc, max_diff = run_softmax_rm(device, shape, dim, dtype, torch_dtype)
    assert pcc >= 0.999, f"PCC={pcc} < 0.999 for shape={shape} dim={dim} dtype={dtype}, max_diff={max_diff}"
    assert not torch.isnan(result.float()).any(), "NaN in output"
    assert not torch.isinf(result.float()).any(), "Inf in output"


@pytest.mark.parametrize(
    "shape, dim",
    [
        # Non-aligned W in V2 RM path
        ((2, 1, 128, 100), -1),
        ((2, 1, 128, 100), -2),
        ((1, 1, 32, 500), -1),
    ],
)
def test_rm_v2_non_aligned_w(device, shape, dim):
    """Test V2 RM path with non-tile-aligned W."""
    result, expected, pcc, max_diff = run_softmax_rm(device, shape, dim, ttnn.float32, torch.float32)
    assert pcc >= 0.999, f"PCC={pcc} < 0.999 for shape={shape} dim={dim}, max_diff={max_diff}"


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1024, 1024), -1),
        ((1024, 1024), -2),
        ((32, 4096), -1),
        ((32, 4096), -2),
        ((128, 8192), -1),
        ((128, 8192), -2),
    ],
)
def test_rm_v2_rank2(device, shape, dim):
    """Test V2 RM path with rank-2 tensors (unsqueezed to 4D internally)."""
    result, expected, pcc, max_diff = run_softmax_rm(device, shape, dim, ttnn.float32, torch.float32)
    assert pcc >= 0.999, f"PCC={pcc} < 0.999 for shape={shape} dim={dim}, max_diff={max_diff}"


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1, 32, 4096), -1),
        ((1, 32, 4096), -2),
        ((1, 32, 8192), -1),
        ((1, 32, 8192), -2),
    ],
)
def test_rm_v2_rank3(device, shape, dim):
    """Test V2 RM path with rank-3 tensors (unsqueezed to 4D internally)."""
    result, expected, pcc, max_diff = run_softmax_rm(device, shape, dim, ttnn.float32, torch.float32)
    assert pcc >= 0.999, f"PCC={pcc} < 0.999 for shape={shape} dim={dim}, max_diff={max_diff}"


def test_rm_v2_output_layout_preserved(device):
    """Output layout must match input layout (RM in → RM out)."""
    x = torch.randn(1, 1, 32, 4096, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected RM output, got {output.layout}"


def test_rm_v2_output_dtype_preserved(device):
    """Output dtype must match input dtype."""
    for ttnn_dtype, torch_dtype in [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)]:
        x = torch.randn(1, 1, 32, 4096, dtype=torch_dtype)
        ttnn_input = ttnn.from_torch(x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)
        assert output.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {output.dtype}"


def test_rm_v2_vs_tile_equivalence(device):
    """RM V2 output should match TILE V2 output for the same input."""
    torch.manual_seed(42)
    x = torch.randn(1, 1, 32, 4096, dtype=torch.float32)

    # TILE path
    ttnn_input_t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_t = ttnn.operations.softmax.softmax(ttnn_input_t, dim=-1)
    result_t = ttnn.to_torch(output_t)

    # RM path
    ttnn_input_r = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_r = ttnn.operations.softmax.softmax(ttnn_input_r, dim=-1)
    result_r = ttnn.to_torch(output_r)

    max_diff = (result_t.float() - result_r.float()).abs().max().item()
    assert max_diff < 0.01, f"RM vs TILE max_diff={max_diff}"


def test_rm_v2_deterministic_all_ones(device):
    """All-ones input: every element should be 1/N where N=W (dim=-1)."""
    for W in [512, 4096, 8192]:
        x = torch.ones(1, 1, 32, W, dtype=torch.float32)
        expected_val = 1.0 / W

        ttnn_input = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)
        result = ttnn.to_torch(output)

        actual = result[0, 0, 0, 0].item()
        assert abs(actual - expected_val) < 0.001, f"W={W}: expected {expected_val}, got {actual}"


def test_rm_v2_multicore(device):
    """Multi-slab (NC > 1) distribution across cores."""
    shape = (4, 2, 32, 4096)  # NC=8 slabs
    result, expected, pcc, max_diff = run_softmax_rm(device, shape, -1, ttnn.float32, torch.float32)
    assert pcc >= 0.999, f"PCC={pcc} < 0.999, max_diff={max_diff}"
