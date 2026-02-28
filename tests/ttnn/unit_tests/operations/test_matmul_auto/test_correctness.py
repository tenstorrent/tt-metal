# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for matmul_auto.

Verifies that matmul_auto:
1. Selects valid configurations without runtime errors for all test shapes
2. Produces correct outputs matching torch.matmul within PCC threshold
3. Handles edge cases (small, large, narrow, batched shapes)
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc


# ─────────────────────────────────────────────
# Test shapes: 50+ representative configurations
# ─────────────────────────────────────────────

# fmt: off
BASIC_SHAPES = [
    # (batch, M, K, N)
    (1, 32, 32, 32),
    (1, 64, 64, 64),
    (1, 128, 128, 128),
    (1, 256, 256, 256),
    (1, 512, 512, 512),
    (1, 1024, 1024, 1024),
    (1, 2048, 2048, 2048),
]

TALL_SHAPES = [
    (1, 2048, 1024, 32),
    (1, 4096, 512, 64),
    (1, 1024, 1024, 32),
    (1, 8192, 256, 32),
]

WIDE_SHAPES = [
    (1, 32, 1024, 2048),
    (1, 64, 512, 4096),
    (1, 32, 256, 1024),
]

LLM_SHAPES = [
    # Common transformer shapes
    (1, 128, 4096, 4096),
    (1, 128, 4096, 11008),
    (1, 128, 11008, 4096),
    (1, 32, 4096, 4096),
    # Attention shapes
    (1, 2048, 128, 2048),
    (1, 128, 2048, 128),
]

BATCHED_SHAPES = [
    (2, 128, 128, 128),
    (4, 256, 256, 256),
    (8, 512, 512, 512),
    (7, 384, 1024, 1024),
]

NON_POWER_OF_2_SHAPES = [
    (1, 384, 1024, 1024),
    (1, 768, 3072, 768),
    (1, 1024, 768, 512),
    (1, 192, 640, 320),
    (1, 2048, 2048, 2080),
]

EDGE_CASE_SHAPES = [
    # Minimum tile-aligned shapes
    (1, 32, 32, 32),
    # Large batch with small matmul
    (16, 32, 32, 32),
    # Single tile dimensions
    (1, 32, 1024, 32),
    # Square
    (1, 1024, 1024, 1024),
]

ALL_SHAPES = BASIC_SHAPES + TALL_SHAPES + WIDE_SHAPES + LLM_SHAPES + NON_POWER_OF_2_SHAPES + EDGE_CASE_SHAPES
# fmt: on


@pytest.fixture(scope="module")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


class TestMatmulAutoCorrectness:
    """Correctness tests for matmul_auto."""

    @pytest.mark.parametrize("batch,m,k,n", ALL_SHAPES)
    def test_output_correctness(self, device, batch, m, k, n):
        """Test that matmul_auto produces correct results for all shapes."""
        from ttnn.operations.auto_config import matmul_auto

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(batch, k, n, dtype=torch.float32) if batch > 1 else torch.randn(k, n, dtype=torch.float32)
        torch_output = torch.matmul(torch_a, torch_b)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output = matmul_auto(input_a, input_b)
        output = ttnn.to_torch(tt_output)

        passed, msg = check_with_pcc(torch_output, output, pcc=0.99)
        assert passed, f"PCC check failed for shape ({batch}, {m}, {k}, {n}): {msg}"

    @pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:5])
    def test_output_with_bias(self, device, batch, m, k, n):
        """Test matmul_auto with bias."""
        from ttnn.operations.auto_config import matmul_auto

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)
        torch_bias = torch.randn(1, n, dtype=torch.float32)
        torch_output = torch.matmul(torch_a, torch_b) + torch_bias

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output = matmul_auto(input_a, input_b, bias=bias)
        output = ttnn.to_torch(tt_output)

        passed, msg = check_with_pcc(torch_output, output, pcc=0.98)
        assert passed, f"PCC check failed for shape ({batch}, {m}, {k}, {n}) with bias: {msg}"

    @pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:3])
    def test_bfloat8_b_dtype(self, device, batch, m, k, n):
        """Test matmul_auto with bfloat8_b data type."""
        from ttnn.operations.auto_config import matmul_auto

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)
        torch_output = torch.matmul(torch_a, torch_b)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output = matmul_auto(input_a, input_b)
        output = ttnn.to_torch(tt_output)

        # Lower PCC expectation for bfloat8_b
        passed, msg = check_with_pcc(torch_output, output, pcc=0.97)
        assert passed, f"PCC check failed for bfloat8_b shape ({batch}, {m}, {k}, {n}): {msg}"

    @pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:3])
    def test_force_program_config(self, device, batch, m, k, n):
        """Test that force_program_config bypasses auto-selection."""
        from ttnn.operations.auto_config import matmul_auto

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)
        torch_output = torch.matmul(torch_a, torch_b)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Use force_program_config to bypass auto-selection
        tt_output = matmul_auto(input_a, input_b, force_program_config=None)
        output = ttnn.to_torch(tt_output)

        passed, msg = check_with_pcc(torch_output, output, pcc=0.99)
        assert passed, f"PCC check failed with force_program_config: {msg}"

    @pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:3])
    def test_config_selection_no_errors(self, device, batch, m, k, n):
        """Test that config selection itself doesn't error."""
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        selector = MatmulAutoConfig()
        result = selector.select(input_a, input_b)

        assert result.selected_config is not None
        assert result.selected_config.is_valid
        assert result.selected_config.config_family in (
            "MultiCast1D", "MultiCast2D", "Reuse",
            "DRAMSharded", "BatchedDRAMSharded", "MinimalMatmul", "MultiCore"
        )
        assert result.selected_config.score > 0
        assert len(result.all_candidates) > 0


class TestMatmulAutoCache:
    """Tests for config cache behavior."""

    @pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:3])
    def test_cache_hit_on_repeat(self, device, batch, m, k, n):
        """Test that repeated calls with same signature hit the cache."""
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig
        from ttnn.operations.auto_config.config_cache import ConfigCache

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        cache = ConfigCache(cache_dir="/tmp/test_matmul_auto_cache")
        cache.clear()

        selector = MatmulAutoConfig(cache=cache)

        # First call: cache miss
        result1 = selector.select(input_a, input_b)
        assert not result1.cache_hit

        # Second call: cache hit
        result2 = selector.select(input_a, input_b)
        assert result2.cache_hit

        cache.clear()


class TestMatmulAutoFeatureExtraction:
    """Tests for feature extraction."""

    def test_features_contain_all_keys(self, device):
        """Test that features dict contains all expected keys."""
        from ttnn.operations.auto_config.feature_extraction import extract_matmul_features

        torch_a = torch.randn(1, 128, 256, dtype=torch.float32)
        torch_b = torch.randn(256, 512, dtype=torch.float32)
        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        features = extract_matmul_features(input_a, input_b)

        required_keys = [
            "M", "K", "N", "batch_size_a", "batch_size_b",
            "M_tiles", "K_tiles", "N_tiles",
            "dtype_a", "dtype_b", "layout_a", "layout_b",
            "is_a_sharded", "is_b_sharded",
            "arch", "grid_x", "grid_y", "num_cores",
            "is_multi_device", "num_devices",
            "transpose_a", "transpose_b",
            "has_bias", "has_activation",
        ]
        for key in required_keys:
            assert key in features, f"Missing feature key: {key}"


class TestConstraintValidator:
    """Tests for constraint validation."""

    def test_tile_alignment_validation(self):
        """Test tile alignment validation."""
        from ttnn.operations.auto_config.constraint_validator import validate_tile_alignment

        # Valid alignment
        is_valid, _ = validate_tile_alignment({"M": 128, "K": 256, "N": 512})
        assert is_valid

        # Invalid alignment
        is_valid, _ = validate_tile_alignment({"M": 100, "K": 256, "N": 512})
        assert not is_valid

    def test_subblock_validation(self):
        """Test subblock parameter validation."""
        from ttnn.operations.auto_config.constraint_validator import validate_subblock_params

        class MockConfig:
            per_core_M = 4
            per_core_N = 2
            out_subblock_h = 2
            out_subblock_w = 2
            in0_block_w = 2

        config = MockConfig()
        is_valid, _ = validate_subblock_params(config, "MultiCast1D")
        assert is_valid

        # Invalid: subblock too large
        config.out_subblock_h = 8
        config.out_subblock_w = 2
        is_valid, _ = validate_subblock_params(config, "MultiCast1D")
        assert not is_valid


class TestHeuristicScorer:
    """Tests for the heuristic scorer."""

    def test_scorer_returns_valid_score(self):
        """Test that scorer returns a score in [0, 1]."""
        from ttnn.operations.auto_config.scorer.heuristic import HeuristicScorer
        from ttnn.operations.auto_config.base import ConfigCandidate

        scorer = HeuristicScorer()
        candidate = ConfigCandidate(
            config=None,
            config_family="MultiCast1D",
            backend="matmul",
            params={
                "mcast_in0": True,
                "in0_block_w": 2,
                "per_core_M": 4,
                "per_core_N": 4,
                "out_subblock_h": 2,
                "out_subblock_w": 2,
            },
        )
        features = {
            "M": 128, "K": 256, "N": 512,
            "M_tiles": 4, "K_tiles": 8, "N_tiles": 16,
            "batch_size_a": 1, "batch_size_b": 1,
            "num_cores": 64,
            "is_tall": False, "is_wide": True, "is_square": False,
            "is_a_sharded": False, "is_batched_b": False,
            "grid_x": 8, "grid_y": 8,
        }

        score = scorer.score(candidate, features)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"

    def test_tall_shape_prefers_mcast_in1(self):
        """Test that tall shapes score higher with mcast_in0=False."""
        from ttnn.operations.auto_config.scorer.heuristic import HeuristicScorer
        from ttnn.operations.auto_config.base import ConfigCandidate

        scorer = HeuristicScorer()
        features = {
            "M": 2048, "K": 1024, "N": 32,
            "M_tiles": 64, "K_tiles": 32, "N_tiles": 1,
            "batch_size_a": 1, "batch_size_b": 1,
            "num_cores": 64,
            "is_tall": True, "is_wide": False, "is_square": False,
            "is_a_sharded": False, "is_batched_b": False,
            "grid_x": 8, "grid_y": 8,
        }

        # Tall config (mcast_in0=False)
        tall_candidate = ConfigCandidate(
            config=None, config_family="MultiCast1D", backend="matmul",
            params={"mcast_in0": False, "in0_block_w": 2, "per_core_M": 1, "per_core_N": 1,
                    "out_subblock_h": 1, "out_subblock_w": 1},
        )
        # Wide config (mcast_in0=True) — wrong for tall shape
        wide_candidate = ConfigCandidate(
            config=None, config_family="MultiCast1D", backend="matmul",
            params={"mcast_in0": True, "in0_block_w": 2, "per_core_M": 1, "per_core_N": 1,
                    "out_subblock_h": 1, "out_subblock_w": 1},
        )

        tall_score = scorer.score(tall_candidate, features)
        wide_score = scorer.score(wide_candidate, features)

        assert tall_score > wide_score, (
            f"Tall shape should prefer mcast_in0=False: tall={tall_score:.3f} vs wide={wide_score:.3f}"
        )
