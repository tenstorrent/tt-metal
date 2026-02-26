# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for mixed_tile assignment using real ttnn BFP quantization."""

from __future__ import annotations

import torch
from loguru import logger

from models.demos.deepseek_v3_b1.mixed_tile.assigner import MixedTileAssigner
from models.demos.deepseek_v3_b1.mixed_tile.metrics import metric_value
from models.demos.deepseek_v3_b1.mixed_tile.tile_utils import (
    MIXED_TILE_BYTES_PER_ELEM,
    MIXED_TILE_FORMATS,
    ttnn_quantize_fn,
)

ALL_BFP_FORMATS = ["bfp8", "bfp4", "bfp2"]


def test_bfp2_outputs_are_powers_of_two():
    """BFP2 = 1 sign bit + 1 mantissa bit + shared exponent per face row.

    The mantissa bit acts as a presence flag:
      - mantissa=0 → value is 0.0
      - mantissa=1 → value is ±2^shared_exp

    During decode, the leading 1 is stripped (IEEE 754 implicit bit), leaving
    no fractional mantissa bits. So every non-zero output is exactly ±2^n.
    """
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    q = ttnn_quantize_fn(x, "bfp2")
    nonzero = q[q != 0]
    log2_vals = torch.log2(torch.abs(nonzero))
    logger.info(f"Non-zero count: {nonzero.numel()} / {q.numel()}")
    assert torch.allclose(log2_vals, log2_vals.round(), atol=1e-5)


def test_bfp2_preserves_signs():
    """BFP2 sign bit must match the original value's sign for all non-zero outputs."""
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    q = ttnn_quantize_fn(x, "bfp2")

    mask = q != 0
    logger.info(f"Non-zero values: {mask.sum().item()} / {q.numel()}")
    assert (torch.sign(q[mask]) == torch.sign(x[mask])).all()


def test_bfp2_bfp4_pcc_comparison():
    """More mantissa bits → higher precision.

    BFP2 (1-bit mantissa) gives ~0.88 PCC on random data.
    BFP4 (3-bit mantissa) gives ~0.99 PCC on random data.
    Both should be positively correlated, and BFP4 >= BFP2.
    """
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    pcc2 = metric_value(x.numpy(), ttnn_quantize_fn(x, "bfp2").numpy(), "pcc")
    pcc4 = metric_value(x.numpy(), ttnn_quantize_fn(x, "bfp4").numpy(), "pcc")
    logger.info(f"bfp2 PCC: {pcc2:.6f}, bfp4 PCC: {pcc4:.6f}")
    assert pcc2 > 0.85, f"bfp2 PCC {pcc2:.4f} unexpectedly low (expected ~0.88)"
    assert pcc4 > 0.98, f"bfp4 PCC {pcc4:.4f} unexpectedly low (expected ~0.99)"
    assert pcc4 >= pcc2, f"bfp4 PCC {pcc4:.4f} < bfp2 PCC {pcc2:.4f}"


def test_smooth_data_picks_cheap_format():
    """On smooth data, the assigner should pick the cheapest format (bfp2)."""
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 0.01

    assigner = MixedTileAssigner(metric="pcc", threshold=0.8, formats=ALL_BFP_FORMATS)
    result = assigner.assign(x, ttnn_quantize_fn)
    logger.info(f"Tile counts: {result.tile_counts}")
    assert result.tile_counts["bfp2"] > 0, "Expected some tiles assigned to bfp2"


def test_hard_tiles_promote_to_bfp8():
    """With a tight threshold (0.995), all tiles need bfp8 since bfp4 PCC is ~0.993.

    With a looser threshold (0.89), bfp2 can handle easy tiles but hard ones
    still need bfp4 or bfp8, so we get a mix of all three formats.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 64)

    # Tight threshold: bfp4 PCC (~0.993) is below 0.995, so everything goes to bfp8
    assigner_tight = MixedTileAssigner(metric="pcc", threshold=0.995, formats=ALL_BFP_FORMATS)
    result_tight = assigner_tight.assign(x, ttnn_quantize_fn)
    logger.info(f"Tight (0.995) tile counts: {result_tight.tile_counts}")
    assert (
        result_tight.tile_counts["bfp8"] == 4
    ), f"All 4 tiles should be bfp8 at threshold 0.995: {result_tight.tile_counts}"

    # Loose threshold: bfp2 can handle some tiles
    assigner_loose = MixedTileAssigner(metric="pcc", threshold=0.89, formats=ALL_BFP_FORMATS)
    result_loose = assigner_loose.assign(x, ttnn_quantize_fn)
    logger.info(f"Loose (0.89) tile counts: {result_loose.tile_counts}")
    cheap_tiles = result_loose.tile_counts["bfp2"] + result_loose.tile_counts["bfp4"]
    assert cheap_tiles > 0, f"Expected some cheap tiles at threshold 0.89: {result_loose.tile_counts}"


def test_mixed_tile_assignment_with_bfp2():
    """Mixed-tile should assign easy tiles to bfp2 and hard tiles to bfp4/bfp8.

    One tile has high dynamic range (linspace -100..100) which bfp2 can't handle
    well, so it should get promoted. The remaining smooth tiles should stay at bfp2.
    """
    torch.manual_seed(42)
    x = torch.randn(128, 128)
    x[:32, :32] = torch.linspace(-100, 100, 32 * 32).reshape(32, 32)
    threshold = 0.89

    assigner = MixedTileAssigner(metric="pcc", threshold=threshold, formats=ALL_BFP_FORMATS)
    result = assigner.assign(x, ttnn_quantize_fn)

    logger.info(f"Tile counts: {result.tile_counts}")
    logger.info(f"Assignment map:\n{result.assignment}")

    # Should have bfp2 for easy tiles and something more expensive for the hard tile
    assert result.tile_counts["bfp2"] > 0, f"No tiles assigned to bfp2: {result.tile_counts}"
    promoted = result.tile_counts["bfp4"] + result.tile_counts["bfp8"]
    assert promoted > 0, f"No tiles promoted above bfp2: {result.tile_counts}"

    # Check per-tile PCC for bfp2 tiles meets the threshold
    fmt_to_idx = {fmt: idx for idx, fmt in enumerate(MIXED_TILE_FORMATS)}
    tile_hw = 32
    xn = x.numpy()
    qn = result.quantized.numpy()
    tiles_h, tiles_w = xn.shape[0] // tile_hw, xn.shape[1] // tile_hw
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            if result.assignment[tr, tc] == fmt_to_idx["bfp2"]:
                ref_tile = xn[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
                q_tile = qn[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
                tile_pcc = metric_value(ref_tile, q_tile, "pcc")
                logger.info(f"  bfp2 tile ({tr},{tc}): PCC={tile_pcc:.6f}")
                assert tile_pcc >= threshold, f"bfp2 tile ({tr},{tc}) PCC {tile_pcc:.4f} below threshold {threshold}"

    # Check overall tensor PCC
    overall_pcc = metric_value(xn, qn, "pcc")
    logger.info(f"Overall mixed-tile PCC: {overall_pcc:.6f}")
    assert overall_pcc >= threshold, f"Overall PCC {overall_pcc:.4f} below threshold {threshold}"


def test_mixed_tile_pcc_above_threshold():
    """The reconstructed tensor from mixed-tile should meet the global quality threshold."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    threshold = 0.85
    assigner = MixedTileAssigner(metric="pcc", threshold=threshold, formats=ALL_BFP_FORMATS)
    result = assigner.assign(x, ttnn_quantize_fn)

    pcc = metric_value(x.numpy(), result.quantized.numpy(), "pcc")
    logger.info(f"Tile counts: {result.tile_counts}")
    logger.info(f"Global PCC: {pcc:.6f} (threshold: {threshold:.4f})")
    assert pcc >= threshold * 0.999, f"Global PCC {pcc:.6f} below threshold {threshold}"


def test_mixed_tile_better_than_uniform_bfp2():
    """Mixed bfp8/bfp4/bfp2 should achieve equal or better quality than uniform bfp2."""
    torch.manual_seed(7)
    x = torch.randn(128, 128)
    x[:32, :32] *= 50.0  # outlier tiles to make uniform bfp2 struggle

    assigner = MixedTileAssigner(metric="pcc", threshold=0.89, formats=ALL_BFP_FORMATS)
    result = assigner.assign(x, ttnn_quantize_fn)

    pcc_mixed = metric_value(x.numpy(), result.quantized.numpy(), "pcc")
    pcc_uniform = metric_value(x.numpy(), ttnn_quantize_fn(x, "bfp2").numpy(), "pcc")

    logger.info(f"Tile counts: {result.tile_counts}")
    logger.info(f"Mixed PCC: {pcc_mixed:.6f}, Uniform bfp2 PCC: {pcc_uniform:.6f}")
    assert pcc_mixed >= pcc_uniform, f"Mixed PCC ({pcc_mixed:.6f}) should be >= uniform bfp2 PCC ({pcc_uniform:.6f})"


def test_mixed_tile_cheaper_than_uniform_bfp8():
    """Mixed bfp8/bfp4/bfp2 should use fewer bytes than uniform bfp8."""
    torch.manual_seed(99)
    x = torch.randn(128, 128)

    assigner = MixedTileAssigner(metric="pcc", threshold=0.85, formats=ALL_BFP_FORMATS)
    result = assigner.assign(x, ttnn_quantize_fn)

    num_tiles = 16  # 128/32 * 128/32
    uniform_bfp8_bytes = num_tiles * 1024 * MIXED_TILE_BYTES_PER_ELEM["bfp8"]

    logger.info(f"Tile counts: {result.tile_counts}")
    logger.info(f"Mixed bytes: {result.total_bytes:.0f}, Uniform bfp8 bytes: {uniform_bfp8_bytes:.0f}")
    assert (
        result.total_bytes <= uniform_bfp8_bytes
    ), f"Mixed bytes ({result.total_bytes:.0f}) should be <= uniform bfp8 ({uniform_bfp8_bytes:.0f})"


def test_tighter_threshold_uses_more_expensive_formats():
    """A tighter threshold should assign more tiles to expensive formats."""
    torch.manual_seed(123)
    x = torch.randn(128, 128)
    x[:32, :32] *= 20.0  # make one region harder

    loose = MixedTileAssigner(metric="pcc", threshold=0.8, formats=ALL_BFP_FORMATS)
    tight = MixedTileAssigner(metric="pcc", threshold=0.999, formats=ALL_BFP_FORMATS)

    result_loose = loose.assign(x, ttnn_quantize_fn)
    result_tight = tight.assign(x, ttnn_quantize_fn)

    logger.info(f"Loose: {result_loose.tile_counts}")
    logger.info(f"Tight: {result_tight.tile_counts}")

    # Tight threshold should use fewer bfp2 tiles (more promoted to bfp4/bfp8)
    assert result_tight.tile_counts["bfp2"] <= result_loose.tile_counts["bfp2"], (
        f"Tight should use <= bfp2 tiles: tight={result_tight.tile_counts['bfp2']}, "
        f"loose={result_loose.tile_counts['bfp2']}"
    )
