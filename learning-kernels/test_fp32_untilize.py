# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FP32 Untilize Precision Tests
==============================

These tests verify whether untilize preserves exact FP32 precision across
different tensor widths and code paths. They are designed to measure progress
on GitHub issue #33795.

Run all tests:
    pytest learning-kernels/test_fp32_untilize.py -v

Run a specific test:
    pytest learning-kernels/test_fp32_untilize.py::test_fast_path_fp32 -v

Run with output showing precision details:
    pytest learning-kernels/test_fp32_untilize.py -v -s
"""

import pytest
import torch
import ttnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def precision_report(expected: torch.Tensor, actual: torch.Tensor) -> str:
    """Return a human-readable precision comparison."""
    if torch.equal(expected, actual):
        return "EXACT MATCH"

    diff = (expected.float() - actual.float()).abs()
    nonzero_mask = diff > 0
    n_wrong = nonzero_mask.sum().item()
    n_total = expected.numel()

    lines = [
        f"  Mismatched elements: {n_wrong}/{n_total} ({100*n_wrong/n_total:.2f}%)",
        f"  Max absolute error:  {diff.max().item():.10e}",
        f"  Mean absolute error: {diff[nonzero_mask].mean().item():.10e}" if n_wrong > 0 else "",
    ]

    # Show a few example mismatches
    flat_diff = diff.flatten()
    worst_indices = flat_diff.topk(min(5, n_wrong)).indices
    for idx in worst_indices:
        i = idx.item()
        lines.append(
            f"    [{i}] expected={expected.flatten()[i].item():.10e}  "
            f"got={actual.flatten()[i].item():.10e}  "
            f"diff={flat_diff[i].item():.10e}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test: Fast path (pack_untilize) - should already work after PR #33904
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],  # 1 tile wide
        [1, 1, 32, 64],  # 2 tiles wide
        [1, 1, 32, 128],  # 4 tiles wide
        [1, 1, 32, 256],  # 8 tiles wide (MAX_PACK_UNTILIZE_WIDTH)
    ],
    ids=["1tile", "2tile", "4tile", "8tile"],
)
def test_fast_path_fp32(device, shape):
    """
    Verify FP32 precision on the fast pack_untilize path (width <= 8 tiles).
    These should pass - PR #33904 fixed this.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.float32)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize(tile_tensor, use_pack_untilize=True)
    result = ttnn.to_torch(untilized)

    report = precision_report(torch_tensor, result)
    print(f"\n  Fast path {shape}: {report}")
    assert torch.equal(result, torch_tensor), f"Fast path lost precision:\n{report}"


# ---------------------------------------------------------------------------
# Test: Slow path (untilize) - THE BUG. These currently fail.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],  # 1 tile wide (slow path forced)
        [1, 1, 32, 64],  # 2 tiles wide
        [1, 1, 32, 256],  # 8 tiles wide
        [1, 1, 32, 512],  # 16 tiles wide
    ],
    ids=["1tile", "2tile", "8tile", "16tile"],
)
def test_slow_path_fp32_precision(device, shape):
    """
    Test FP32 precision when forcing the slow untilize path.
    Currently EXPECTED TO FAIL (TF32 truncation).
    When your fix works, remove the xfail marker.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.float32)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize(tile_tensor, use_pack_untilize=False)
    result = ttnn.to_torch(untilized)

    report = precision_report(torch_tensor, result)
    print(f"\n  Slow path {shape}: {report}")
    assert torch.equal(result, torch_tensor), f"Slow path lost FP32 precision:\n{report}"


# ---------------------------------------------------------------------------
# Test: Wide FP32 tensors - the real-world case
# These are too wide for pack_untilize and currently fall back to slow path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 320],  # 10 tiles wide (just over MAX_PACK_UNTILIZE_WIDTH)
        [1, 1, 32, 512],  # 16 tiles wide
        [1, 1, 32, 1024],  # 32 tiles wide
        [1, 1, 64, 512],  # 2 rows of 16-tile blocks
        [1, 1, 128, 7328],  # Real-world wide tensor (from issue tests)
    ],
    ids=["10tile", "16tile", "32tile", "2row_16tile", "realworld_wide"],
)
def test_wide_fp32_untilize(device, shape):
    """
    Wide FP32 tensors that exceed MAX_PACK_UNTILIZE_WIDTH=8.
    These are the main target of this fix.
    Currently EXPECTED TO FAIL. When your fix works, these pass.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.float32)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize(tile_tensor)  # default path selection
    result = ttnn.to_torch(untilized)

    report = precision_report(torch_tensor, result)
    print(f"\n  Wide FP32 {shape}: {report}")
    assert torch.equal(result, torch_tensor), f"Wide FP32 untilize lost precision:\n{report}"


# ---------------------------------------------------------------------------
# Test: untilize_with_unpadding - same issue, different op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,output_end",
    [
        ([1, 1, 128, 7328], [0, 0, 119, 7299]),  # Commented out in existing tests
        ([4128, 512], [4127, 511]),  # Also commented out
        ([1, 1, 64, 320], [0, 0, 55, 310]),  # Moderate size
    ],
    ids=["wide_7328", "tall_4128", "moderate"],
)
def test_wide_fp32_untilize_with_unpadding(device, shape, output_end):
    """
    Wide FP32 untilize_with_unpadding. Same underlying issue as untilize.
    Currently EXPECTED TO FAIL.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.float32)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize_with_unpadding(tile_tensor, output_tensor_end=output_end)
    result = ttnn.to_torch(untilized)

    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    expected = torch_tensor[slices]

    report = precision_report(expected, result)
    print(f"\n  Unpadding FP32 {shape}->{output_end}: {report}")
    assert torch.equal(result, expected), f"untilize_with_unpadding lost FP32 precision:\n{report}"


# ---------------------------------------------------------------------------
# Test: BFloat16 control - should always pass (not affected by this bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 512],
        [1, 1, 128, 1024],
    ],
    ids=["16tile", "32tile_tall"],
)
def test_bfloat16_unaffected(device, shape):
    """
    BFloat16 untilize should always work. This is a control test to make sure
    your changes don't break anything.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.float32).to(torch.bfloat16)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize(tile_tensor)
    result = ttnn.to_torch(untilized)

    assert torch.equal(result, torch_tensor), "BFloat16 untilize broke - your changes may have a regression"


# ---------------------------------------------------------------------------
# Diagnostic: Measure precision loss quantitatively (always runs, never fails)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width_tiles",
    [1, 4, 8, 10, 16, 32],
)
def test_measure_precision_loss(device, width_tiles):
    """
    Diagnostic test - does NOT assert, just prints precision info.
    Use this to understand the magnitude of the TF32 truncation.
    """
    shape = [1, 1, 32, width_tiles * 32]
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.float32)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize(tile_tensor)
    result = ttnn.to_torch(untilized)

    exact = torch.equal(result, torch_tensor)
    report = precision_report(torch_tensor, result)
    print(f"\n  Width={width_tiles} tiles ({width_tiles*32} elements): {'EXACT' if exact else 'LOSSY'}")
    if not exact:
        print(report)
