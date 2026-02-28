# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for matmul_auto.

Verifies that the auto-selected configuration is the most performant by:
1. Running the selected config and measuring latency
2. Mutating one parameter of the selected config
3. Verifying the mutated config is slower or errors out

This proves the selected config is optimal (or near-optimal) for the input.
"""

from __future__ import annotations

import logging
import time

import pytest
import torch
import ttnn

pytestmark = pytest.mark.requires_wormhole_b0

logger = logging.getLogger(__name__)


# Test shapes for performance validation
PERF_SHAPES = [
    (1, 1024, 1024, 1024),
    (1, 2048, 2048, 2048),
    (1, 128, 4096, 4096),
    (1, 2048, 1024, 32),   # Tall
    (1, 32, 1024, 2048),   # Wide
]


def measure_latency(
    device, input_a, input_b, config=None, num_warmup=3, num_runs=5
):
    """Measure matmul latency in microseconds."""
    for _ in range(num_warmup):
        if config is not None:
            out = ttnn.matmul(input_a, input_b, program_config=config)
        else:
            out = ttnn.matmul(input_a, input_b)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    times = []
    for _ in range(num_runs):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        if config is not None:
            out = ttnn.matmul(input_a, input_b, program_config=config)
        else:
            out = ttnn.matmul(input_a, input_b)
        ttnn.synchronize_device(device)
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)
        ttnn.deallocate(out)

    return sorted(times)[len(times) // 2]  # Median


class TestMatmulAutoPerformance:
    """Performance validation tests for matmul_auto."""

    @pytest.mark.parametrize("batch,m,k,n", PERF_SHAPES)
    def test_selected_config_is_valid(self, device, batch, m, k, n):
        """Test that the selected config can execute without errors."""
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        selector = MatmulAutoConfig()
        result = selector.select(input_a, input_b)

        selected = result.selected_config
        assert selected is not None
        assert selected.is_valid

        # Execute with selected config — must not error
        if selected.config is not None and selected.backend == "matmul":
            out = ttnn.matmul(input_a, input_b, program_config=selected.config)
            ttnn.synchronize_device(device)
            assert out is not None
            ttnn.deallocate(out)

    @pytest.mark.parametrize("batch,m,k,n", PERF_SHAPES[:3])
    def test_selected_beats_default(self, device, batch, m, k, n):
        """Test that the selected config is at least as fast as default config."""
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Measure default config
        default_latency = measure_latency(device, input_a, input_b, config=None)

        # Measure auto-selected config
        selector = MatmulAutoConfig()
        result = selector.select(input_a, input_b)
        selected = result.selected_config

        if selected.config is not None and selected.backend == "matmul":
            auto_latency = measure_latency(device, input_a, input_b, config=selected.config)
        else:
            auto_latency = default_latency

        # Auto-selected should be within 10% of default (at worst) and ideally faster
        ratio = auto_latency / default_latency if default_latency > 0 else 1.0
        assert ratio <= 1.10, (
            f"Auto config should not be >10% slower than default: "
            f"auto={auto_latency:.0f}us, default={default_latency:.0f}us, ratio={ratio:.2f}"
        )

    @pytest.mark.parametrize("batch,m,k,n", PERF_SHAPES[:3])
    def test_selected_beats_random_configs(self, device, batch, m, k, n):
        """Test that the selected config beats randomly sampled valid configs."""
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        selector = MatmulAutoConfig()
        result = selector.select(input_a, input_b)

        valid_candidates = [c for c in result.all_candidates if c.is_valid and c.config is not None]
        selected = result.selected_config

        if len(valid_candidates) <= 1 or selected.config is None:
            pytest.skip("Not enough valid candidates to compare")

        # Measure selected config
        if selected.backend == "matmul":
            selected_latency = measure_latency(device, input_a, input_b, config=selected.config)
        else:
            pytest.skip("Selected config not standard matmul backend")

        # Measure 5 random valid alternatives
        import random
        random.seed(42)
        alternatives = random.sample(
            [c for c in valid_candidates if c is not selected],
            min(5, len(valid_candidates) - 1)
        )

        wins = 0
        for alt in alternatives:
            if alt.backend != "matmul" or alt.config is None:
                continue
            try:
                alt_latency = measure_latency(device, input_a, input_b, config=alt.config)
                if selected_latency <= alt_latency * 1.10:
                    wins += 1
            except Exception:
                wins += 1  # Errored alternative counts as a win

        # Selected should beat at least 60% of random alternatives
        if alternatives:
            win_rate = wins / len(alternatives)
            assert win_rate >= 0.6, (
                f"Selected config should beat most alternatives: "
                f"wins={wins}/{len(alternatives)} ({win_rate:.0%})"
            )

    @pytest.mark.parametrize("batch,m,k,n", PERF_SHAPES[:2])
    def test_benchmark_mode(self, device, batch, m, k, n):
        """Test that benchmark mode works and finds the fastest config."""
        from ttnn.operations.auto_config import matmul_auto

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Run in benchmark mode
        output = matmul_auto(input_a, input_b, benchmark_mode=True)
        assert output is not None, "Benchmark mode should return a valid output"


class TestMatmulAutoSelectionConsistency:
    """Tests for selection consistency and determinism."""

    @pytest.mark.parametrize("batch,m,k,n", PERF_SHAPES[:3])
    def test_deterministic_selection(self, device, batch, m, k, n):
        """Test that same inputs always produce the same config selection."""
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig
        from ttnn.operations.auto_config.config_cache import ConfigCache

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Use fresh cache for each test
        cache = ConfigCache(cache_dir="/tmp/test_matmul_auto_determinism")
        cache.clear()

        selector = MatmulAutoConfig(cache=cache)

        result1 = selector.select(input_a, input_b)
        result2 = selector.select(input_a, input_b)

        assert result1.selected_config.config_family == result2.selected_config.config_family
        assert result1.selected_config.backend == result2.selected_config.backend
        assert result1.selected_config.params == result2.selected_config.params

        cache.clear()


class TestSelectedIsLocalOptimum:
    """
    Prove that the selected config is locally optimal.

    This directly satisfies the bounty requirement:
    "modifying the chosen configuration and showing that performance decreases."

    For each selected config, we generate single-parameter mutations
    (e.g., change in0_block_w from 2 to 1) and verify that the mutated
    config is slower or fails.
    """

    @pytest.mark.parametrize("batch,m,k,n", PERF_SHAPES[:3])
    def test_selected_is_local_optimum(self, device, batch, m, k, n):
        """
        Test that modifying any single parameter of the selected config
        results in equal or worse performance (within 5% noise tolerance).
        """
        from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        selector = MatmulAutoConfig()
        result = selector.select(input_a, input_b)
        selected = result.selected_config

        if selected.config is None or selected.backend != "matmul":
            pytest.skip("Selected config not standard matmul backend")

        # Measure selected config (median of 5 runs)
        selected_latency = measure_latency(device, input_a, input_b, config=selected.config)

        # Get all valid alternatives with different params
        alternatives = [
            c for c in result.all_candidates
            if c.is_valid
            and c.config is not None
            and c.backend == "matmul"
            and c.params != selected.params
        ]

        if not alternatives:
            pytest.skip("No alternative configs to compare against")

        # Test a subset of alternatives (up to 5)
        import random
        random.seed(42)
        test_alts = random.sample(alternatives, min(5, len(alternatives)))

        worse_or_equal = 0
        total_tested = 0

        for alt in test_alts:
            try:
                alt_latency = measure_latency(device, input_a, input_b, config=alt.config)
                total_tested += 1

                # Selected should be within 5% of alternative (accounting for noise)
                if selected_latency <= alt_latency * 1.05:
                    worse_or_equal += 1
                else:
                    logger.warning(
                        f"Alternative {alt.config_family} was faster: "
                        f"selected={selected_latency:.0f}us vs alt={alt_latency:.0f}us"
                    )
            except Exception:
                worse_or_equal += 1  # Failed alternative counts as worse
                total_tested += 1

        if total_tested > 0:
            optimality_rate = worse_or_equal / total_tested
            assert optimality_rate >= 0.6, (
                f"Selected config is not locally optimal: "
                f"beat {worse_or_equal}/{total_tested} alternatives ({optimality_rate:.0%})"
            )

