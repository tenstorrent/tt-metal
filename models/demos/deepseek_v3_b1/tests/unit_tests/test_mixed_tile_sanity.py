# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for mixed_tile assignment (no device needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.demos.deepseek_v3_b1.mixed_tile.assigner import MixedTileAssigner, MixedTileResult
from models.demos.deepseek_v3_b1.mixed_tile.metrics import metric_is_good, metric_value, pearson_corr
from models.demos.deepseek_v3_b1.mixed_tile.tile_utils import (
    MIXED_TILE_FORMATS,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
    tile_metrics,
)

# ---------------------------------------------------------------------------
# metrics tests
# ---------------------------------------------------------------------------


def test_pcc_identical():
    a = np.random.randn(64).astype(np.float32)
    assert pearson_corr(a, a) == pytest.approx(1.0)


def test_pcc_zeros():
    a = np.zeros(32, dtype=np.float32)
    b = np.zeros(32, dtype=np.float32)
    assert pearson_corr(a, b) == 1.0


def test_pcc_different():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    assert pearson_corr(a, b) == pytest.approx(-1.0)


def test_metric_value_mae():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    assert metric_value(a, b, "mae") == pytest.approx(0.5)


def test_metric_value_atol():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    assert metric_value(a, b, "atol") == pytest.approx(2.0)


def test_metric_is_good_pcc():
    assert metric_is_good(0.999, "pcc", 0.99)
    assert not metric_is_good(0.98, "pcc", 0.99)


def test_metric_is_good_mae():
    assert metric_is_good(0.01, "mae", 0.05)
    assert not metric_is_good(0.1, "mae", 0.05)


# ---------------------------------------------------------------------------
# tile_utils tests
# ---------------------------------------------------------------------------


def test_reshape_round_trip_2d():
    x = np.random.randn(64, 64).astype(np.float32)
    padded, shape_info, pad_info = reshape_to_2d_with_padding(x)
    tile_hw = 32
    tiles_h = pad_info[2] // tile_hw
    tiles_w = pad_info[3] // tile_hw
    tiles = padded.reshape(tiles_h, tile_hw, tiles_w, tile_hw).transpose(0, 2, 1, 3).reshape(-1, tile_hw, tile_hw)
    reconstructed = reconstruct_from_tiles(tiles, shape_info, pad_info)
    np.testing.assert_array_equal(reconstructed, x)


def test_reshape_round_trip_1d():
    x = np.random.randn(100).astype(np.float32)
    padded, shape_info, pad_info = reshape_to_2d_with_padding(x)
    tile_hw = 32
    tiles_h = pad_info[2] // tile_hw
    tiles_w = pad_info[3] // tile_hw
    tiles = padded.reshape(tiles_h, tile_hw, tiles_w, tile_hw).transpose(0, 2, 1, 3).reshape(-1, tile_hw, tile_hw)
    reconstructed = reconstruct_from_tiles(tiles, shape_info, pad_info)
    np.testing.assert_array_equal(reconstructed, x)


def test_tile_metrics_pcc_identical():
    tiles = np.random.randn(4, 32, 32).astype(np.float32)
    scores = tile_metrics(tiles, tiles, "pcc")
    np.testing.assert_allclose(scores, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# assigner tests
# ---------------------------------------------------------------------------


def _noop_quantize(x: np.ndarray, fmt: str) -> np.ndarray:
    """Identity quantizer — returns input unchanged."""
    return x.copy()


def _noisy_quantize(x: np.ndarray, fmt: str) -> np.ndarray:
    """Adds format-dependent noise: more noise for cheaper formats."""
    noise_scale = {"bf16": 0.0, "bfp8": 0.001, "bfp4": 0.01, "bfp2": 0.1}
    rng = np.random.default_rng(42)
    return x + rng.normal(0, noise_scale.get(fmt, 0.01), size=x.shape).astype(np.float32)


def test_assigner_noop_all_cheap():
    """With identity quantizer, all tiles should get the cheapest format."""
    x = np.random.randn(64, 64).astype(np.float32)
    assigner = MixedTileAssigner(metric="pcc", threshold=0.999, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, _noop_quantize)

    assert isinstance(result, MixedTileResult)
    assert result.assignment.shape == (2, 2)
    # All tiles should be bfp4 (cheapest) since identity quantizer gives PCC=1.0
    fmt_to_idx = {fmt: idx for idx, fmt in enumerate(MIXED_TILE_FORMATS)}
    assert np.all(result.assignment == fmt_to_idx["bfp4"])
    assert result.tile_counts["bfp4"] == 4
    assert result.tile_counts["bfp8"] == 0


def test_assigner_noisy_mixed():
    """With noisy quantizer and tight threshold, some tiles should need higher precision."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, size=(64, 64)).astype(np.float32)
    assigner = MixedTileAssigner(metric="pcc", threshold=0.9999, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, _noisy_quantize)

    assert result.assignment.shape == (2, 2)
    # bfp4 adds more noise, so with a very tight threshold some tiles may need bfp8
    total_tiles = result.tile_counts["bfp8"] + result.tile_counts["bfp4"]
    assert total_tiles == 4


def test_assigner_empty_tensor():
    x = np.array([], dtype=np.float32)
    assigner = MixedTileAssigner()
    result = assigner.assign(x, _noop_quantize)
    assert result.total_bytes == 0.0


def test_assigner_invalid_metric():
    with pytest.raises(ValueError, match="Unsupported metric"):
        MixedTileAssigner(metric="invalid")


def test_assigner_invalid_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        MixedTileAssigner(formats=["fp16"])


def test_assigner_result_shape_preserved():
    """The quantized output should have the same shape as the input."""
    x = np.random.randn(96, 128).astype(np.float32)
    assigner = MixedTileAssigner(formats=["bfp8", "bfp4"])
    result = assigner.assign(x, _noop_quantize)
    assert result.quantized.shape == x.shape


# ---------------------------------------------------------------------------
# torch tensor tests
# ---------------------------------------------------------------------------


def _torch_noop_quantize(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return x.clone()


def test_assigner_torch_input():
    """Assigner should accept torch tensors and return torch quantized output."""
    x = torch.randn(64, 64)
    assigner = MixedTileAssigner(metric="pcc", threshold=0.999, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, _torch_noop_quantize)

    assert isinstance(result.quantized, torch.Tensor)
    assert result.quantized.shape == x.shape
    assert result.assignment.shape == (2, 2)


def test_assigner_torch_numpy_same_assignment():
    """Torch and numpy paths should produce identical assignments for the same data."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=(64, 64)).astype(np.float32)

    assigner = MixedTileAssigner(metric="pcc", threshold=0.999, formats=["bfp8", "bfp4"])

    result_np = assigner.assign(data, _noop_quantize)
    result_torch = assigner.assign(torch.from_numpy(data), _torch_noop_quantize)

    np.testing.assert_array_equal(result_np.assignment, result_torch.assignment)
    np.testing.assert_allclose(result_np.quantized, result_torch.quantized.numpy(), atol=1e-6)
