# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance gates for the Time Series Transformer demo.

These measure the traced generate path end to end, once the program cache and trace are warm.
They are sensitive to host CPU load: the model is small enough that wall-clock is dominated by
per-op dispatch and host-side tensor assembly rather than device arithmetic, so a contended
machine inflates every number here.

Each test holds at most one traced model at a time (see :func:`traced_model`). Keeping two
alive and replaying them alternately wedges the device.
"""

import time
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.time_series_transformer.reference.torch_reference import (
    compute_metrics,
    generate_mean_reference,
    make_inputs,
)
from models.demos.time_series_transformer.tests.perf.perf_common import (
    STRETCH_BATCH,
    STRETCH_CONTEXT,
    STRETCH_LATENCY_MS,
    STRETCH_SAMPLE_COUNT,
    STRETCH_SAMPLE_SECONDS,
    STRETCH_THROUGHPUT,
    TARGET_LATENCY_MS,
    TARGET_SAMPLE_SECONDS,
    TARGET_THROUGHPUT,
    build_model,
    run_benchmark,
)
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer

THROUGHPUT_BATCHES = (1, 8, 32, 64)

# The optimized profile is held to a looser bound than the 5% parity gate: bfloat16 carries
# 8 mantissa bits, and the flash kernel reorders the softmax accumulation.
OPTIMIZED_MAE_TOLERANCE = 0.10

PROFILES = {
    "accuracy": {},
    "performance": {"dtype": "bfloat16", "use_sdpa": True, "use_exact_softmax": False},
}


@contextmanager
def traced_model(device, config, hf_state, profile: str = "accuracy"):
    """Build a traced model, then release its trace before anything else captures one.

    A trace pins device allocations for its lifetime. Two live traces replayed in turn hang
    the device, so this is a context manager rather than a fixture -- it makes the lifetime
    explicit and non-overlapping.
    """
    model = build_model(replace(config, use_trace=True, **PROFILES[profile]), hf_state, device=device)
    try:
        yield model
    finally:
        model.release_traces()
        ttnn.synchronize_device(device)


@pytest.mark.models_performance_bare_metal
class TestStageOneTargets:
    def test_single_sequence_latency(self, device, config, hf_state, hf_model):
        """Batch 1, one trajectory: the end-to-end forecast latency gate."""
        inputs = make_inputs(hf_model.config, batch=1)
        with traced_model(device, config, hf_state) as model:
            result = run_benchmark(model, inputs, batch=1, num_samples=1, mode="mean")

        logger.info(
            f"latency: {result.latency_ms:.2f} ms "
            f"(Stage 1 target < {TARGET_LATENCY_MS} ms, Stage 3 stretch < {STRETCH_LATENCY_MS} ms)"
        )
        assert (
            result.latency_ms < TARGET_LATENCY_MS
        ), f"batch-1 latency {result.latency_ms:.2f} ms exceeds {TARGET_LATENCY_MS} ms"
        assert (
            result.latency_ms < STRETCH_LATENCY_MS
        ), f"batch-1 latency {result.latency_ms:.2f} ms exceeds the Stage 3 stretch {STRETCH_LATENCY_MS} ms"

    def test_throughput(self, device, config, hf_state, hf_model):
        """Best sustained sequences/second across a batch sweep."""
        best = 0.0
        with traced_model(device, config, hf_state) as model:
            for batch in THROUGHPUT_BATCHES:
                inputs = make_inputs(hf_model.config, batch=batch)
                result = run_benchmark(model, inputs, batch=batch, num_samples=1, mode="mean", iterations=3)
                logger.info(str(result))
                best = max(best, result.throughput)

        logger.info(f"best throughput: {best:.1f} seq/s (target >= {TARGET_THROUGHPUT})")
        assert best >= TARGET_THROUGHPUT, f"throughput {best:.1f} seq/s below {TARGET_THROUGHPUT}"

    def test_sample_generation(self, device, config, hf_state, hf_model):
        """100 trajectories for one series, drawn as a single batched rollout."""
        inputs = make_inputs(hf_model.config, batch=1)
        with traced_model(device, config, hf_state) as model:
            result = run_benchmark(model, inputs, batch=1, num_samples=100, mode="sample", iterations=3)

        seconds = result.latency_ms / 1000.0
        logger.info(f"100 samples: {seconds:.3f} s (target < {TARGET_SAMPLE_SECONDS} s)")
        assert seconds < TARGET_SAMPLE_SECONDS, f"100 samples took {seconds:.3f} s"


@pytest.mark.models_performance_bare_metal
class TestTraceEquivalence:
    def test_traced_matches_eager(self, device, config, hf_state, hf_model):
        """Trace replay must reproduce the eager rollout."""
        inputs = make_inputs(hf_model.config, batch=2)
        eager = build_model(replace(config, use_trace=False), hf_state, device=device)
        eager_output = eager.generate(num_parallel_samples=1, mode="mean", **inputs)

        with traced_model(device, config, hf_state) as model:
            traced_output = model.generate(num_parallel_samples=1, mode="mean", **inputs)

        _, _, pcc = compute_metrics(eager_output, traced_output)
        logger.info(f"traced vs eager PCC: {pcc:.8f}")
        assert pcc > 0.9999, f"traced path diverged from eager: PCC {pcc:.8f}"

        # The two paths are mathematically identical but not bit-identical: the cached eager
        # path accumulates one token at a time while the traced path recomputes the window,
        # so device float32 rounding lands differently. Bound the relative gap instead.
        relative = (traced_output - eager_output).abs() / eager_output.abs().clamp_min(1e-6)
        logger.info(f"max relative difference: {float(relative.max()):.2e}")
        assert float(relative.max()) < 1e-3

    @pytest.mark.parametrize("mode", ["mean", "sample"])
    def test_trace_is_reused_across_calls(self, device, config, hf_state, hf_model, mode):
        """A second call at the same shape must not recapture.

        Mean mode runs the whole rollout from one trace and so registers a rollout runner;
        sampling still steps through the decoder and registers a per-step runner.
        """
        inputs = make_inputs(hf_model.config, batch=1)
        expected_kind = "rollout" if mode == "mean" else "decode"

        with traced_model(device, config, hf_state) as model:
            model.generate(num_parallel_samples=1, mode=mode, **inputs)
            assert model._trace_key == (expected_kind, 1), f"{mode} mode captured the wrong trace"

            before = model._trace_runner
            model.generate(num_parallel_samples=1, mode=mode, **inputs)
            assert model._trace_runner is before, "generate recaptured a trace for a known shape"


@pytest.mark.models_performance_bare_metal
class TestOptimizedProfile:
    """Stage 2/3 profile: what bfloat16 and the flash-attention kernel cost and buy."""

    def test_sdpa_accepts_padded_head_dim(self, device, config, hf_state, hf_model):
        """head_dim=13 is not tile-aligned; the kernel needs it zero-padded to 32."""
        inputs = make_inputs(hf_model.config, batch=1)
        with traced_model(device, config, hf_state, profile="performance") as model:
            assert model.config.use_sdpa
            output = model.generate(num_parallel_samples=1, mode="mean", **inputs)
        assert torch.isfinite(output).all()

    def test_accuracy_against_reference(self, device, config, hf_state, hf_model):
        inputs = make_inputs(hf_model.config, batch=4)
        reference = generate_mean_reference(hf_model, inputs)
        with traced_model(device, config, hf_state, profile="performance") as model:
            actual = model.generate(num_parallel_samples=1, mode="mean", **inputs).squeeze(1)

        mse, mae, pcc = compute_metrics(reference, actual)
        relative_mae = mae / float(reference.abs().mean())
        logger.info(f"optimized profile: PCC {pcc:.6f}, relative MAE {relative_mae * 100:.2f}%")
        assert relative_mae < OPTIMIZED_MAE_TOLERANCE, f"relative MAE {relative_mae * 100:.2f}% (mse={mse:.3e})"

    def test_throughput(self, device, config, hf_state, hf_model):
        """Throughput of the optimized profile, for comparison with the accuracy profile."""
        best = 0.0
        with traced_model(device, config, hf_state, profile="performance") as model:
            for batch in (32, 64):
                inputs = make_inputs(hf_model.config, batch=batch)
                result = run_benchmark(model, inputs, batch=batch, num_samples=1, mode="mean", iterations=3)
                logger.info(f"optimized {result}")
                best = max(best, result.throughput)

        logger.info(f"optimized best throughput: {best:.1f} seq/s (stretch >= {STRETCH_THROUGHPUT})")
        assert best >= STRETCH_THROUGHPUT, f"optimized throughput {best:.1f} seq/s below {STRETCH_THROUGHPUT}"


@pytest.mark.models_performance_bare_metal
class TestScalingLimits:
    """Stage 3 scaling targets: sample count, batch width, context length."""

    def test_thousand_samples(self, device, config, hf_state, hf_model):
        """1000 trajectories for one series in under 2 s.

        Only the performance profile clears this. At 1000 rows the cost is device throughput
        at width, not host overhead -- float32 takes 3.43 s, bfloat16 with the flash kernel
        1.77 s.
        """
        inputs = make_inputs(hf_model.config, batch=1)
        with traced_model(device, config, hf_state, profile="performance") as model:
            model.generate(num_parallel_samples=STRETCH_SAMPLE_COUNT, mode="sample", **inputs)
            start = time.perf_counter()
            output = model.generate(num_parallel_samples=STRETCH_SAMPLE_COUNT, mode="sample", **inputs)
            seconds = time.perf_counter() - start

        logger.info(f"{STRETCH_SAMPLE_COUNT} samples: {seconds:.3f} s (target < {STRETCH_SAMPLE_SECONDS} s)")
        assert output.shape == (1, STRETCH_SAMPLE_COUNT, hf_model.config.prediction_length)
        assert torch.isfinite(output).all()
        assert seconds < STRETCH_SAMPLE_SECONDS, f"{STRETCH_SAMPLE_COUNT} samples took {seconds:.3f} s"

    def test_wide_batch(self, device, config, hf_state, hf_model):
        """Forecast many series in one call -- the '100+ time series' target."""
        batch = STRETCH_BATCH
        inputs = make_inputs(hf_model.config, batch=batch)
        with traced_model(device, config, hf_state) as model:
            result = run_benchmark(model, inputs, batch=batch, num_samples=1, mode="mean", iterations=3)

        logger.info(f"batch {batch}: {result.latency_ms:.2f} ms, {result.throughput:.1f} seq/s")
        assert result.throughput >= TARGET_THROUGHPUT

    def test_long_context(self, device, config, hf_state, hf_model):
        """The architecture must run at the full stretch context length.

        No checkpoint exists at that size, so this uses untrained weights and checks that the
        shapes hold and the output stays finite -- op shapes do not depend on weight values, so
        the latency is still representative. Static categorical features are dropped because
        their embedding tables come from the checkpoint.
        """
        long_config = replace(
            config,
            context_length=STRETCH_CONTEXT,
            num_static_categorical_features=0,
            cardinality=(),
            embedding_dimension=(),
            feature_size=None,
            use_trace=True,
        )
        model = TimeSeriesTransformer(long_config, device=device)
        try:
            inputs = long_context_inputs(long_config, batch=1)
            model.generate(num_parallel_samples=1, mode="mean", **inputs)
            start = time.perf_counter()
            output = model.generate(num_parallel_samples=1, mode="mean", **inputs)
            latency_ms = (time.perf_counter() - start) * 1000
        finally:
            model.release_traces()
            ttnn.synchronize_device(device)

        logger.info(f"context {STRETCH_CONTEXT}: {latency_ms:.2f} ms")
        assert output.shape == (1, 1, long_config.prediction_length)
        assert torch.isfinite(output).all()


def long_context_inputs(config, batch: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    length = config.past_length
    return {
        "past_values": torch.rand(batch, length, generator=generator) * 100 + 50,
        "past_time_features": torch.randn(batch, length, config.num_time_features, generator=generator),
        "past_observed_mask": torch.ones(batch, length),
        "future_time_features": torch.randn(
            batch, config.prediction_length, config.num_time_features, generator=generator
        ),
        "static_real_features": torch.zeros(batch, config.num_static_real_features),
    }


@pytest.mark.models_performance_bare_metal
class TestTraceLifecycle:
    """Only one trace may be live at a time, whatever sequence of shapes a caller asks for."""

    def test_single_live_trace_across_shape_changes(self, device, config, hf_state, hf_model):
        """A batch sweep changes the row count; each change must release before recapturing.

        Holding several live traces makes tt-metal warn that buffers allocated afterwards may
        be corrupted once a trace executes, which invalidates any measurement taken from them.
        """
        with traced_model(device, config, hf_state) as model:
            for batch in (1, 8, 32):
                inputs = make_inputs(hf_model.config, batch=batch)
                output = model.generate(num_parallel_samples=1, mode="mean", **inputs)
                assert torch.isfinite(output).all()
                assert model._trace_key == ("rollout", batch)

            # Switching mode swaps the trace rather than adding a second one.
            model.generate(num_parallel_samples=2, mode="sample", **make_inputs(hf_model.config, batch=1))
            assert model._trace_key == ("decode", 2)

    def test_release_clears_the_live_trace(self, device, config, hf_state, hf_model):
        with traced_model(device, config, hf_state) as model:
            model.generate(num_parallel_samples=1, mode="mean", **make_inputs(hf_model.config, batch=1))
            assert model._trace_runner is not None
            model.release_traces()
            assert model._trace_runner is None and model._trace_key is None
