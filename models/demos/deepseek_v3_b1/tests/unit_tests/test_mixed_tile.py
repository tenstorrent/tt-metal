# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for mixed_tile assignment using real ttnn BFP quantization."""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_b1.mixed_tile.assigner import MixedTileAssigner
from models.demos.deepseek_v3_b1.mixed_tile.metrics import metric_value
from models.demos.deepseek_v3_b1.mixed_tile.tile_utils import MIXED_TILE_BYTES_PER_ELEM, MIXED_TILE_FORMATS

# ---------------------------------------------------------------------------
# ttnn quantize-dequantize helpers
# ---------------------------------------------------------------------------

TTNN_DTYPE_MAP = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
}


def ttnn_quantize_fn(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantize-dequantize round trip through ttnn."""
    tt_dtype = TTNN_DTYPE_MAP[fmt]
    tt_tensor = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt_tensor).to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_assignment_prefers_cheap_format_when_quality_is_met():
    """On smooth data, bfp4 should be sufficient for most tiles at a reasonable threshold."""
    torch.manual_seed(0)
    # Smooth data (small values, low variance) → bfp4 should handle it fine
    x = torch.randn(64, 64) * 0.01

    assigner = MixedTileAssigner(metric="pcc", threshold=0.99, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)

    assert result.tile_counts["bfp4"] > 0, "Expected at least some tiles assigned to bfp4"
    total = result.tile_counts["bfp4"] + result.tile_counts["bfp8"]
    assert total == 4  # 64x64 = 2x2 tiles


def test_assignment_promotes_to_expensive_format_for_hard_tiles():
    """Tiles with high dynamic range should get promoted to bfp8 under MAE metric."""
    torch.manual_seed(0)
    # Tile (0,0): mix of large and medium values — bfp4 (3-bit mantissa) will have
    # significant absolute error compared to bfp8 (7-bit mantissa)
    x = torch.randn(64, 64)
    x[:32, :32] = torch.linspace(-50, 50, 32 * 32).reshape(32, 32)
    # Other tiles: near-zero, trivial for any format
    x[32:, :] *= 0.001
    x[:32, 32:] *= 0.001

    # Use MAE with a tight threshold — bfp4 error on the linspace tile should exceed it
    assigner = MixedTileAssigner(metric="mae", threshold=0.05, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)

    fmt_to_idx = {fmt: idx for idx, fmt in enumerate(MIXED_TILE_FORMATS)}
    # The linspace tile should need bfp8; the near-zero tiles should be fine with bfp4
    assert (
        result.assignment[0, 0] == fmt_to_idx["bfp8"]
    ), f"Tile (0,0) with high dynamic range should be bfp8, got {MIXED_TILE_FORMATS[result.assignment[0, 0]]}"
    # At least one other tile should be bfp4 (the near-zero tiles)
    assert result.tile_counts["bfp4"] > 0, "Expected some tiles to remain bfp4"


def test_quantized_output_meets_quality_threshold():
    """The reconstructed tensor from mixed-tile should meet the global quality threshold."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    threshold = 0.999
    assigner = MixedTileAssigner(metric="pcc", threshold=threshold, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)

    pcc = metric_value(x.numpy(), result.quantized.numpy(), "pcc")
    assert pcc >= threshold * 0.999, f"Global PCC {pcc:.6f} below threshold {threshold}"


def test_mixed_tile_better_than_uniform_cheap():
    """Mixed-tile should achieve equal or better quality than uniform bfp4."""
    torch.manual_seed(7)
    x = torch.randn(128, 128)
    # Add some outlier tiles to make uniform bfp4 struggle
    x[:32, :32] *= 50.0

    assigner = MixedTileAssigner(metric="pcc", threshold=0.999, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)

    uniform_bfp4 = ttnn_quantize_fn(x, "bfp4")
    pcc_mixed = metric_value(x.numpy(), result.quantized.numpy(), "pcc")
    pcc_uniform = metric_value(x.numpy(), uniform_bfp4.numpy(), "pcc")

    assert (
        pcc_mixed >= pcc_uniform
    ), f"Mixed-tile PCC ({pcc_mixed:.6f}) should be >= uniform bfp4 PCC ({pcc_uniform:.6f})"


def test_mixed_tile_cheaper_than_uniform_expensive():
    """Mixed-tile should use fewer bytes than uniform bfp8 when some tiles can be bfp4."""
    torch.manual_seed(99)
    x = torch.randn(128, 128)

    assigner = MixedTileAssigner(metric="pcc", threshold=0.99, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)

    num_tiles = 16  # 128/32 * 128/32
    uniform_bfp8_bytes = num_tiles * 1024 * MIXED_TILE_BYTES_PER_ELEM["bfp8"]

    assert (
        result.total_bytes <= uniform_bfp8_bytes
    ), f"Mixed-tile bytes ({result.total_bytes:.0f}) should be <= uniform bfp8 ({uniform_bfp8_bytes:.0f})"


def test_different_thresholds_different_assignments():
    """A tighter threshold should assign more tiles to expensive formats."""
    torch.manual_seed(123)
    x = torch.randn(128, 128)
    x[:32, :32] *= 20.0  # make one region harder

    loose = MixedTileAssigner(metric="pcc", threshold=0.99, formats=["bfp8", "bfp4"])
    tight = MixedTileAssigner(metric="pcc", threshold=0.9999, formats=["bfp8", "bfp4"])

    result_loose = loose.assign(x, ttnn_quantize_fn)
    result_tight = tight.assign(x, ttnn_quantize_fn)

    assert result_tight.tile_counts["bfp8"] >= result_loose.tile_counts["bfp8"], (
        f"Tight threshold should use >= bfp8 tiles: tight={result_tight.tile_counts['bfp8']}, "
        f"loose={result_loose.tile_counts['bfp8']}"
    )
