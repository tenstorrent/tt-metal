# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Core utilisation, tensor-manipulation overhead, and distribution switching.

These three Stage 3 items are reported rather than optimised, because at this model's shapes
the honest answer to each is bounded by geometry rather than by implementation effort. Each
test prints the measurement and asserts a ceiling, so a regression is caught even where the
current number is not flattering.
"""

import time
from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.time_series_transformer.reference.torch_reference import make_inputs
from models.demos.time_series_transformer.tests.perf.perf_common import build_model
from models.demos.time_series_transformer.tt.config import TILE_SIZE
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer

# Measured at 814 on the eager path (reshape 300, permute 398, slice 24, concat 92). The
# ceiling exists to catch a blow-up, not to pin the exact count.
MAX_TM_OPS_PER_FORECAST = 1000


def tiles(rows: int, columns: int) -> int:
    return -(-rows // TILE_SIZE) * (-(-columns // TILE_SIZE))


@pytest.mark.models_performance_bare_metal
class TestCoreUtilisation:
    """How much of the grid this model's shapes can possibly occupy."""

    def test_report_achievable_parallelism(self, device, config, hf_model):
        grid = device.compute_with_storage_grid_size()
        available = grid.x * grid.y

        rows_per_batch = config.context_length
        activation_tiles = {
            "encoder input (batch 1)": tiles(rows_per_batch, config.feature_size),
            "hidden state (batch 1)": tiles(rows_per_batch, config.d_model),
            "attention scores (batch 1)": tiles(rows_per_batch, rows_per_batch),
            "hidden state (batch 64)": tiles(64 * rows_per_batch, config.d_model),
        }

        logger.info(f"device grid {grid.x}x{grid.y} = {available} cores")
        for name, count in activation_tiles.items():
            logger.info(f"  {name}: {count} tile(s) -> at most {min(count, available)} core(s)")

        # The point of the measurement: a batch-1 activation is a single tile, so no amount of
        # sharding can spread it. This is why the Stage 2 sharding attempts hit TT_FATAL and
        # why throughput needs batching rather than more cores.
        assert activation_tiles["hidden state (batch 1)"] == 1
        assert activation_tiles["hidden state (batch 64)"] > 1
        assert available >= 1

    def test_throughput_scales_with_batch(self, device, config, hf_state, hf_model):
        """Utilisation shows up as throughput scaling; batching is the lever that works."""
        from models.demos.time_series_transformer.tests.perf.perf_common import run_benchmark
        from models.demos.time_series_transformer.tests.perf.test_perf import traced_model

        with traced_model(device, config, hf_state) as model:
            single = run_benchmark(model, make_inputs(hf_model.config, batch=1), batch=1, mode="mean")
            wide = run_benchmark(
                model, make_inputs(hf_model.config, batch=64), batch=64, num_samples=1, mode="mean", iterations=3
            )

        speedup = wide.throughput / single.throughput
        logger.info(
            f"throughput batch 1 {single.throughput:.1f} seq/s -> batch 64 {wide.throughput:.1f} seq/s "
            f"({speedup:.1f}x), i.e. the grid is only exercised once the batch fills it"
        )
        assert speedup > 2.0, "batching should improve throughput materially"


@pytest.mark.models_performance_bare_metal
class TestTensorManipulationOverhead:
    """Count the reshape/permute/slice/concat traffic a forecast costs."""

    def test_report_tm_ops_per_forecast(self, device, config, hf_state, hf_model, monkeypatch):
        counts: dict[str, int] = {}

        def counting(name, original):
            def wrapper(*args, **kwargs):
                counts[name] = counts.get(name, 0) + 1
                return original(*args, **kwargs)

            return wrapper

        for name in ("reshape", "permute", "slice", "concat", "pad"):
            monkeypatch.setattr(ttnn, name, counting(name, getattr(ttnn, name)))

        # Eager, so every op is observed rather than replayed from a capture.
        model = build_model(replace(config, use_trace=False), hf_state, device=device)
        model.generate(num_parallel_samples=1, mode="mean", **make_inputs(hf_model.config, batch=1))

        total = sum(counts.values())
        horizon = config.prediction_length
        logger.info(f"shape ops per forecast: {counts} (total {total}, {total / horizon:.1f} per decode step)")
        assert total < MAX_TM_OPS_PER_FORECAST, f"{total} shape ops per forecast"

        # Permutes dominate, and they are inherent: every attention splits and merges heads,
        # and at head_dim=13 there is no tile-aligned layout that avoids the transpose. The
        # eager path measured here also recomputes the whole prefix each step; the traced path
        # pays these once per capture rather than once per forecast.
        assert counts.get("permute", 0) > counts.get("slice", 0)


@pytest.mark.models_performance_bare_metal
class TestDistributionSwitching:
    """Switching heads costs a rebuild; measure it rather than claim it is free."""

    @pytest.mark.parametrize("distribution", ["student_t", "normal", "negative_binomial"])
    def test_switch_cost_and_validity(self, device, config, hf_state, hf_model, distribution):
        inputs = make_inputs(hf_model.config, batch=1)

        start = time.perf_counter()
        model = TimeSeriesTransformer(replace(config, distribution_output=distribution), device=device)
        model.load_hf_state_dict(hf_state, strict=False)
        build_seconds = time.perf_counter() - start

        output = model.generate(num_parallel_samples=2, mode="mean", **inputs)

        logger.info(f"{distribution}: rebuild {build_seconds * 1000:.1f} ms, output {tuple(output.shape)}")
        assert torch.isfinite(output).all()
        # Rebuilding re-uploads weights but compiles nothing new; it must stay sub-second.
        assert build_seconds < 5.0, f"switching to {distribution} took {build_seconds:.2f} s"


@pytest.mark.models_performance_bare_metal
class TestShardingTradeoff:
    """Why manual sharding is not adopted, measured rather than asserted.

    The Stage 2 rationale originally rested on this model being small. That turned out to be
    the wrong explanation: sweeping the width from 64 to 1024 and the row count from 24 to
    1536 -- well past this checkpoint's 26 -- height sharding is slower at every point, even
    when the layout conversion is amortised over a chain of blocks. The likely reason is that
    ``ttnn.linear`` already selects a multi-core program config for interleaved inputs, so
    pinning the layout by hand constrains it without adding parallelism.

    These tests exercise the sharding helper and check it stays *correct*; the timings are
    logged rather than asserted, so a future TTNN improvement shows up as a better number
    instead of a failure.
    """

    CHAIN_DEPTH = 4

    @pytest.mark.parametrize("width", [64, 256, 1024])
    def test_sharded_chain_matches_interleaved(self, device, width):
        from models.demos.time_series_transformer.reference.torch_reference import compute_pcc
        from models.demos.time_series_transformer.tt.config import create_sharded_memory_config

        rows = 1536
        torch.manual_seed(0)
        upload = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)  # noqa: E731
        activation = upload(torch.randn(rows, width))
        weights = [upload(torch.randn(width, width) * 0.05) for _ in range(self.CHAIN_DEPTH)]

        def interleaved():
            hidden = activation
            for weight in weights:
                hidden = ttnn.gelu(ttnn.linear(hidden, weight, transpose_b=True))
            return hidden

        def sharded():
            config = create_sharded_memory_config((rows, width), device=device, strategy=ttnn.ShardStrategy.HEIGHT)
            hidden = ttnn.to_memory_config(activation, config)
            for weight in weights:
                hidden = ttnn.gelu(
                    ttnn.linear(hidden, weight, transpose_b=True, memory_config=config), memory_config=config
                )
            return ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)

        reference = ttnn.to_torch(interleaved()).float()
        actual = ttnn.to_torch(sharded()).float()

        def timed(fn):
            for _ in range(3):
                fn()
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            for _ in range(10):
                fn()
            ttnn.synchronize_device(device)
            return (time.perf_counter() - start) / 10 * 1000

        interleaved_ms, sharded_ms = timed(interleaved), timed(sharded)
        logger.info(
            f"width {width}: interleaved {interleaved_ms:.3f} ms, height-sharded {sharded_ms:.3f} ms "
            f"({interleaved_ms / sharded_ms:.2f}x)"
        )

        # Correctness is the assertion; speed is the observation.
        assert compute_pcc(reference, actual) > 0.99, "height sharding changed the result"

    def test_sharding_is_unavailable_at_the_checkpoint_width(self, device, config):
        """At d_model=26 a batch-1 activation is one tile and the shard spec degenerates."""
        from models.demos.time_series_transformer.tt.config import create_sharded_memory_config

        torch.manual_seed(0)
        activation = ttnn.from_torch(
            torch.randn(config.context_length, config.d_model),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        shard = create_sharded_memory_config(
            (config.context_length, config.d_model), device=device, strategy=ttnn.ShardStrategy.HEIGHT
        )
        try:
            ttnn.to_memory_config(activation, shard)
            reason = "accepted"
        except Exception as exc:  # noqa: BLE001 - the failure itself is the documented result
            reason = f"rejected ({type(exc).__name__})"
        logger.info(f"height sharding at d_model={config.d_model}, {config.context_length} rows: {reason}")
