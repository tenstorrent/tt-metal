# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from typing import Callable, TypeVar

import pytest
import torch

import ttnn
from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config
from models.demos.wormhole.patchtst.demo.runner import _build_trace_session, run_patchtst
from models.demos.wormhole.patchtst.reference.hf_reference import reference_forward
from models.demos.wormhole.patchtst.tests.helpers import compute_classification_metrics, compute_metrics, prepare_run
from models.demos.wormhole.patchtst.tt.common import (
    DEFAULT_L1_SMALL_SIZE,
    DEFAULT_NUM_COMMAND_QUEUES,
    DEFAULT_TRACE_REGION_SIZE,
    PatchTSTRuntimePolicy,
    resolve_runtime_policy_for_workload,
)
from models.demos.wormhole.patchtst.tt.model import PatchTSTTTNNModel
from models.demos.wormhole.patchtst.tt.streaming import CachedForecastStreamer, HostFullRerunPatchTSTStreamer

PRIMARY_THROUGHPUT_SPS = 150.0
PRIMARY_LATENCY_MS = 40.0
PRIMARY_CORRELATION = 0.90
QUALITY_DELTA_RATIO = 0.05
STRETCH_THROUGHPUT_SPS = 800.0
STRETCH_LATENCY_MS = 15.0
T = TypeVar("T")


def _release_if_possible(value: T | None) -> None:
    release = getattr(value, "release", None)
    if callable(release):
        release()


def _run_for_timing(fn: Callable[[], T], warmup_iterations: int, measurement_iterations: int) -> tuple[float, T]:
    for _ in range(max(warmup_iterations, 0)):
        _release_if_possible(fn())
    output = None
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(max(measurement_iterations, 1)):
            _release_if_possible(output)
            output = fn()
        elapsed = time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed / max(measurement_iterations, 1), output


def _run_median_timing(
    fn: Callable[[], T],
    warmup_iterations: int,
    measurement_iterations: int,
    sample_count: int,
    followup_warmup_iterations: int = 0,
) -> tuple[float, T, list[float]]:
    samples_seconds: list[float] = []
    outputs: list[T] = []
    for sample_idx in range(max(sample_count, 1)):
        sample_warmup = warmup_iterations if sample_idx == 0 else max(followup_warmup_iterations, 0)
        sample_seconds, output = _run_for_timing(
            fn=fn,
            warmup_iterations=sample_warmup,
            measurement_iterations=measurement_iterations,
        )
        samples_seconds.append(sample_seconds)
        outputs.append(output)
    order = sorted(range(len(samples_seconds)), key=lambda idx: samples_seconds[idx])
    median_index = order[len(order) // 2]
    for idx, output in enumerate(outputs):
        if idx != median_index:
            _release_if_possible(output)
    return samples_seconds[median_index], outputs[median_index], samples_seconds


def _cfg(task: str = "forecast", **overrides) -> PatchTSTDemoConfig:
    defaults = {"task": task, "strict_fallback": True, "dataset": "etth1", "batch_size": 1, "max_windows": 64}
    defaults.update(overrides)
    return merge_demo_config(PatchTSTDemoConfig(task=defaults["task"]), **defaults)


def _measure_trace_run(
    cfg: PatchTSTDemoConfig,
    *,
    runtime_policy: PatchTSTRuntimePolicy | None = None,
    measurement_iterations: int = 20,
) -> tuple[float, float]:
    prepared = prepare_run(cfg)
    runtime_policy = resolve_runtime_policy_for_workload(
        policy=runtime_policy or PatchTSTRuntimePolicy(),
        dataset_num_channels=int(prepared.past.shape[-1]),
    )
    ttnn.CONFIG.throw_exception_on_fallback = True
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=DEFAULT_L1_SMALL_SIZE,
        trace_region_size=DEFAULT_TRACE_REGION_SIZE,
        num_command_queues=DEFAULT_NUM_COMMAND_QUEUES,
    )
    try:
        model = PatchTSTTTNNModel(
            demo_config=prepared.runtime_cfg,
            reference=prepared.reference,
            classification_reference=prepared.classification_reference,
            device=device,
            runtime_policy=runtime_policy,
        )
        try:
            session = _build_trace_session(model, prepared.past, prepared.observed, prepared.runtime_cfg.task)
            try:
                seconds, _, _ = _run_median_timing(
                    fn=session.replay,
                    warmup_iterations=4,
                    measurement_iterations=measurement_iterations,
                    sample_count=3,
                    followup_warmup_iterations=2,
                )
                return seconds * 1000.0, prepared.runtime_cfg.batch_size / max(seconds, 1e-9)
            finally:
                session.release()
        finally:
            model.close()
    finally:
        ttnn.close_device(device)


def _measure_wall_ms(fn) -> tuple[float, object]:
    start = time.perf_counter()
    output = fn()
    return (time.perf_counter() - start) * 1000.0, output


def _measure_stream_steps(
    cfg: PatchTSTDemoConfig,
    *,
    stream_steps: int,
    streaming_mode: str,
) -> tuple[float, list[torch.Tensor]]:
    prepared = prepare_run(cfg)
    runtime_policy = PatchTSTRuntimePolicy()
    runtime_policy = resolve_runtime_policy_for_workload(
        policy=runtime_policy,
        dataset_num_channels=int(prepared.past.shape[-1]),
    )
    stream_values = prepared.future[:, :stream_steps, :] if prepared.future is not None else None
    if stream_values is None:
        raise ValueError("Streaming perf checks require forecast data with future values.")
    ttnn.CONFIG.throw_exception_on_fallback = True
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=DEFAULT_L1_SMALL_SIZE,
        trace_region_size=DEFAULT_TRACE_REGION_SIZE,
        num_command_queues=DEFAULT_NUM_COMMAND_QUEUES,
    )
    try:
        model = PatchTSTTTNNModel(
            demo_config=prepared.runtime_cfg,
            reference=prepared.reference,
            device=device,
            runtime_policy=runtime_policy,
        )
        try:
            if streaming_mode == "cached":
                streamer = CachedForecastStreamer(
                    model=model,
                    initial_context=prepared.past,
                    initial_observed_mask=prepared.observed,
                    use_trace=bool(prepared.runtime_cfg.use_trace),
                )
            else:
                streamer = HostFullRerunPatchTSTStreamer(
                    model=model,
                    initial_context=prepared.past,
                    initial_observed_mask=prepared.observed,
                )
            try:
                start = time.perf_counter()
                predictions = []
                for step_idx in range(stream_steps):
                    tick = stream_values[:, step_idx : step_idx + 1, :]
                    predictions.append(streamer.step(tick))
                return (time.perf_counter() - start) * 1000.0, predictions
            finally:
                close = getattr(streamer, "close", None)
                if callable(close):
                    close()
        finally:
            model.close()
    finally:
        ttnn.close_device(device)


def _run_primary_forecast_three_times_in_subprocess() -> list[float]:
    samples: list[float] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    program = """
from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config
from models.demos.wormhole.patchtst.tests.perf.test_patchtst_perf import _measure_trace_run
cfg = merge_demo_config(PatchTSTDemoConfig(task='forecast'), task='forecast', dataset='etth1', batch_size=1, max_windows=64, use_trace=True)
_, throughput = _measure_trace_run(cfg)
print(throughput)
"""
    for _ in range(3):
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=os.getcwd(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        for line in reversed(completed.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(float(line))
                break
            except ValueError:
                continue
        else:
            raise AssertionError(
                "failed to find throughput payload in subprocess output:\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
    return samples


def _forecast_quality_sweep(
    cfg: PatchTSTDemoConfig,
    *,
    batch_size: int,
    max_windows: int,
    runtime_policy: PatchTSTRuntimePolicy | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
    tt_predictions: list[torch.Tensor] = []
    reference_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for offset in range(0, max_windows, batch_size):
        current_batch = min(batch_size, max_windows - offset)
        row_cfg = merge_demo_config(cfg, batch_size=current_batch, window_offset=offset)
        prepared = prepare_run(row_cfg)
        tt_predictions.append(run_patchtst(row_cfg, runtime_policy=runtime_policy))
        reference_predictions.append(
            reference_forward(
                artifacts=prepared.reference,
                past_values=prepared.past,
                future_values=prepared.future,
                target_values=prepared.target_values,
                past_observed_mask=prepared.observed,
            )
        )
        targets.append(prepared.future)

    tt_prediction = torch.cat(tt_predictions, dim=0)
    reference_prediction = torch.cat(reference_predictions, dim=0)
    target = torch.cat(targets, dim=0)
    quality = compute_metrics(tt_prediction, target)
    reference_quality = compute_metrics(reference_prediction, target)
    return (
        compute_metrics(tt_prediction, reference_prediction),
        quality,
        reference_quality,
        int(tt_prediction.shape[0]),
    )


def _forecast_metrics(
    cfg: PatchTSTDemoConfig,
    *,
    runtime_policy: PatchTSTRuntimePolicy | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    prepared = prepare_run(cfg)
    tt_prediction = run_patchtst(cfg, runtime_policy=runtime_policy)
    reference_prediction = reference_forward(
        artifacts=prepared.reference,
        past_values=prepared.past,
        future_values=prepared.future,
        target_values=prepared.target_values,
        past_observed_mask=prepared.observed,
    )
    return (
        compute_metrics(tt_prediction, reference_prediction),
        compute_metrics(tt_prediction, prepared.future),
        compute_metrics(reference_prediction, prepared.future),
    )


@pytest.mark.timeout(1200)
@pytest.mark.models_performance_bare_metal
def test_primary_forecast_trace_batch1_reproducibility():
    cfg = _cfg(task="forecast", dataset="etth1", batch_size=1, max_windows=64, use_trace=True)
    samples = _run_primary_forecast_three_times_in_subprocess()
    print(f"primary_forecast_trace_batch1_reproducibility_sps={samples}")
    assert len(samples) == 3
    assert min(samples) >= PRIMARY_THROUGHPUT_SPS


@pytest.mark.timeout(600)
@pytest.mark.models_performance_bare_metal
def test_primary_forecast_trace_batch1_targets():
    cfg = _cfg(task="forecast", dataset="etth1", batch_size=1, max_windows=64, use_trace=True)
    latency_ms, throughput_sps = _measure_trace_run(cfg)
    parity, quality, reference_quality = _forecast_metrics(cfg)
    mse_delta_ratio = abs(quality["mse"] - reference_quality["mse"]) / max(float(reference_quality["mse"]), 1e-9)
    mae_delta_ratio = abs(quality["mae"] - reference_quality["mae"]) / max(float(reference_quality["mae"]), 1e-9)
    print(f"primary_forecast_trace_batch1 latency_ms={latency_ms:.4f} throughput_sps={throughput_sps:.4f}")
    assert throughput_sps >= PRIMARY_THROUGHPUT_SPS
    assert latency_ms <= PRIMARY_LATENCY_MS
    assert parity["correlation"] >= PRIMARY_CORRELATION
    assert mse_delta_ratio <= QUALITY_DELTA_RATIO
    assert mae_delta_ratio <= QUALITY_DELTA_RATIO
    assert abs(quality["correlation"] - reference_quality["correlation"]) <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
def test_batch16_trace_targets():
    cfg = _cfg(task="forecast", dataset="etth1", batch_size=16, max_windows=128, use_trace=True)
    runtime_policy = PatchTSTRuntimePolicy(activation_memory_tier="l1", sdpa_q_chunk_size=64, sdpa_k_chunk_size=64)
    latency_ms, throughput_sps = _measure_trace_run(cfg, runtime_policy=runtime_policy, measurement_iterations=50)
    parity, quality, reference_quality = _forecast_metrics(cfg, runtime_policy=runtime_policy)
    mse_delta_ratio = abs(quality["mse"] - reference_quality["mse"]) / max(float(reference_quality["mse"]), 1e-9)
    mae_delta_ratio = abs(quality["mae"] - reference_quality["mae"]) / max(float(reference_quality["mae"]), 1e-9)
    print(f"batch16_trace latency_ms={latency_ms:.4f} throughput_sps={throughput_sps:.4f}")
    assert throughput_sps >= STRETCH_THROUGHPUT_SPS
    assert latency_ms <= STRETCH_LATENCY_MS
    assert parity["correlation"] >= PRIMARY_CORRELATION
    assert mse_delta_ratio <= QUALITY_DELTA_RATIO
    assert mae_delta_ratio <= QUALITY_DELTA_RATIO
    assert abs(quality["correlation"] - reference_quality["correlation"]) <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(1200)
@pytest.mark.models_device_performance_bare_metal
def test_batch16_trace_repeatability():
    cfg = _cfg(task="forecast", dataset="etth1", batch_size=16, max_windows=128, use_trace=True)
    runtime_policy = PatchTSTRuntimePolicy(activation_memory_tier="l1", sdpa_q_chunk_size=64, sdpa_k_chunk_size=64)
    first = _measure_trace_run(cfg, runtime_policy=runtime_policy, measurement_iterations=50)
    second = _measure_trace_run(cfg, runtime_policy=runtime_policy, measurement_iterations=50)
    print(f"batch16_trace_repeatability first_sps={first[1]:.4f} second_sps={second[1]:.4f}")
    for latency_ms, throughput_sps in (first, second):
        assert throughput_sps >= STRETCH_THROUGHPUT_SPS
        assert latency_ms <= STRETCH_LATENCY_MS


@pytest.mark.timeout(600)
@pytest.mark.models_performance_bare_metal
def test_sharded_attention_evidence_and_quality():
    cfg = _cfg(
        task="forecast",
        dataset="etth1",
        split="test",
        batch_size=4,
        context_length=512,
        prediction_length=96,
        max_windows=64,
        use_trace=False,
    )
    runtime_policy = PatchTSTRuntimePolicy(
        activation_memory_tier="dram",
        use_sharded_attention_inputs=True,
        use_device_patching=True,
    )
    parity, quality, reference_quality, windows = _forecast_quality_sweep(
        cfg,
        batch_size=4,
        max_windows=64,
        runtime_policy=runtime_policy,
    )
    mse_delta_ratio = abs(quality["mse"] - reference_quality["mse"]) / max(float(reference_quality["mse"]), 1e-9)
    mae_delta_ratio = abs(quality["mae"] - reference_quality["mae"]) / max(float(reference_quality["mae"]), 1e-9)
    print(
        "sharded_attention "
        f"windows={windows} parity_corr={parity['correlation']:.6f} quality_corr={quality['correlation']:.6f}"
    )
    assert windows == 64
    assert parity["correlation"] >= PRIMARY_CORRELATION
    assert mse_delta_ratio <= QUALITY_DELTA_RATIO
    assert mae_delta_ratio <= QUALITY_DELTA_RATIO
    assert abs(quality["correlation"] - reference_quality["correlation"]) <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
def test_cached_streaming_speedup_and_parity():
    cfg = _cfg(task="forecast", dataset="etth1", batch_size=1, max_windows=8, use_trace=True)
    cached_ms, cached = _measure_stream_steps(cfg, stream_steps=8, streaming_mode="cached")
    full_ms, full = _measure_stream_steps(cfg, stream_steps=8, streaming_mode="full-rerun")
    speedup = max((full_ms - cached_ms) / max(full_ms, 1e-9), 0.0)
    print(f"cached_streaming stream_ms={cached_ms:.4f} full_ms={full_ms:.4f} speedup={speedup:.6f}")
    assert speedup > 0.0
    for cached_step, full_step in zip(cached, full, strict=True):
        assert compute_metrics(cached_step, full_step)["correlation"] >= 0.99


@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
def test_shared_encoder_multitask_latency_gain():
    multitask_cfg = _cfg(
        task="multi_task",
        dataset="heartbeat_cls",
        split="test",
        batch_size=4,
        max_windows=8,
        context_length=309,
        prediction_length=96,
    )
    forecast_cfg = _cfg(
        task="forecast",
        dataset="heartbeat_cls",
        split="test",
        batch_size=4,
        max_windows=8,
        context_length=309,
        prediction_length=96,
        checkpoint_id_override="models/demos/wormhole/patchtst/artifacts/finetune/forecast_heartbeat_multitask_ckpt",
        checkpoint_revision_override="local-generated",
    )
    classification_cfg = _cfg(
        task="classification",
        dataset="heartbeat_cls",
        split="test",
        batch_size=4,
        max_windows=8,
        context_length=309,
        prediction_length=96,
        checkpoint_id_override="models/demos/wormhole/patchtst/artifacts/finetune/classification_heartbeat_multitask_ckpt",
        checkpoint_revision_override="local-generated",
    )
    multitask_ms, prediction = _measure_wall_ms(lambda: run_patchtst(multitask_cfg))
    forecast_ms, forecast_prediction = _measure_wall_ms(lambda: run_patchtst(forecast_cfg))
    classification_ms, classification_prediction = _measure_wall_ms(lambda: run_patchtst(classification_cfg))
    prepared = prepare_run(multitask_cfg)
    classification_metrics = compute_classification_metrics(prediction["classification"], prepared.target_values)
    forecast_quality = compute_metrics(prediction["forecast"], prepared.future)
    latency_gain = max(
        ((forecast_ms + classification_ms) - multitask_ms) / max(forecast_ms + classification_ms, 1e-9), 0.0
    )
    print(
        "shared_encoder_multitask "
        f"latency_gain={latency_gain:.6f} "
        f"forecast_corr={compute_metrics(prediction['forecast'], forecast_prediction)['correlation']:.6f} "
        f"classification_acc={classification_metrics['accuracy']:.6f}"
    )
    assert latency_gain > 0.0
    assert compute_metrics(prediction["forecast"], forecast_prediction)["correlation"] >= 0.90
    assert compute_metrics(prediction["classification"], classification_prediction)["correlation"] >= 0.90
    assert forecast_quality["mse"] >= 0.0
    assert classification_metrics["accuracy"] >= 0.0
