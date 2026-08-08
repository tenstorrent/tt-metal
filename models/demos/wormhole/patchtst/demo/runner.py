# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

import ttnn
from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, resolve_runtime_config
from models.demos.wormhole.patchtst.demo.data_utils import build_observed_mask, load_dataset_matrix, load_task_dataset
from models.demos.wormhole.patchtst.reference.hf_reference import (
    adapt_reference_for_runtime_channels,
    adapt_reference_for_runtime_context,
    load_reference_model,
)
from models.demos.wormhole.patchtst.tt.common import (
    DEFAULT_L1_SMALL_SIZE,
    DEFAULT_NUM_COMMAND_QUEUES,
    DEFAULT_TRACE_REGION_SIZE,
    TT_DTYPE,
    PatchTSTRuntimePolicy,
    resolve_runtime_policy_for_workload,
)
from models.demos.wormhole.patchtst.tt.model import PatchTSTTTNNModel
from models.demos.wormhole.patchtst.tt.streaming import CachedForecastStreamer, HostFullRerunPatchTSTStreamer

_HOST_TORCH_THREADS_CONFIGURED = False


@dataclass
class TraceSession:
    replay: Any
    benchmark_replay: Any | None
    materialize: Any
    release: Any


def _configure_host_torch_threads() -> None:
    global _HOST_TORCH_THREADS_CONFIGURED
    if _HOST_TORCH_THREADS_CONFIGURED:
        return
    # PatchTST still does a small amount of host preprocessing before the TT path. One host thread
    # keeps that setup stable without exposing threadpool jitter to normal user runs.
    torch.set_num_threads(1)
    _HOST_TORCH_THREADS_CONFIGURED = True


def _to_torch_prediction(prediction: Any) -> Any:
    if isinstance(prediction, ttnn.Tensor):
        return ttnn.to_torch(ttnn.from_device(prediction))
    if isinstance(prediction, dict):
        return {key: _to_torch_prediction(value) for key, value in prediction.items()}
    return prediction


def _build_trace_session(
    model: PatchTSTTTNNModel,
    past: torch.Tensor,
    observed: torch.Tensor,
    task: str,
) -> TraceSession:
    if task != "forecast":
        raise ValueError("Trace replay is currently supported only for forecasting.")
    if model.cfg.channel_mode != "independent":
        raise ValueError("Trace replay currently supports channel_mode='independent' only.")

    prepared = model.prepare_hidden_input(past_values=past, past_observed_mask=observed)
    hidden_input_addr = prepared.hidden_state.buffer_address()
    hidden_host = ttnn.from_torch(
        ttnn.to_torch(ttnn.from_device(prepared.hidden_state)),
        dtype=TT_DTYPE,
        layout=ttnn.TILE_LAYOUT,
    )

    warm_output = model.forward_from_hidden_tt(prepared, task=task)
    warm_output.release()
    ttnn.synchronize_device(model.device)

    trace_id = ttnn.begin_trace_capture(model.device, cq_id=0)
    traced_output = model.forward_from_hidden_tt(prepared, task=task)
    ttnn.end_trace_capture(model.device, trace_id, cq_id=0)
    ttnn.synchronize_device(model.device)

    if prepared.hidden_state.buffer_address() != hidden_input_addr:
        raise RuntimeError("Trace input buffer address changed after capture; persistent address invariant failed.")

    def _replay():
        ttnn.copy_host_to_device_tensor(hidden_host, prepared.hidden_state, 1)
        ttnn.execute_trace(model.device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(model.device)
        return None

    def _materialize():
        traced_output.prediction = _to_torch_prediction(traced_output.prediction)
        if isinstance(traced_output.prediction, torch.Tensor):
            traced_output.prediction = traced_output.prediction * prepared.scale + prepared.loc
        return traced_output

    def _release():
        ttnn.release_trace(model.device, trace_id)
        prepared.release()
        ttnn.deallocate(hidden_host)

    return TraceSession(replay=_replay, benchmark_replay=None, materialize=_materialize, release=_release)


def run_patchtst(
    demo_config: PatchTSTDemoConfig,
    runtime_policy: PatchTSTRuntimePolicy | None = None,
):
    if ttnn.GetNumAvailableDevices() < 1:
        raise RuntimeError("No Tenstorrent device available.")

    _configure_host_torch_threads()
    if demo_config.task == "multi_task":
        reference = load_reference_model(
            "forecast",
            demo_config.multi_task_checkpoint_ids["forecast"],
            demo_config,
            demo_config.multi_task_checkpoint_revisions["forecast"],
        )
        classification_reference = load_reference_model(
            "classification",
            demo_config.multi_task_checkpoint_ids["classification"],
            demo_config,
            demo_config.multi_task_checkpoint_revisions["classification"],
        )
    else:
        reference = load_reference_model(
            demo_config.task,
            demo_config.effective_checkpoint_id(),
            demo_config,
            demo_config.effective_checkpoint_revision(),
        )
        classification_reference = None
    runtime_cfg = resolve_runtime_config(demo_config, reference.config)
    if int(reference.config.context_length) != int(runtime_cfg.context_length):
        if not runtime_cfg.allow_reference_context_adaptation:
            raise ValueError(
                "Runtime context differs from the checkpoint context, but this run requires a native context-shaped checkpoint. "
                f"runtime_context={runtime_cfg.context_length}, checkpoint_context={reference.config.context_length}, "
                f"checkpoint_id={reference.checkpoint_id}"
            )
        adapt_reference_for_runtime_context(reference, runtime_cfg.context_length)
    task_batch = load_task_dataset(
        runtime_cfg.dataset_root,
        runtime_cfg.dataset,
        runtime_cfg.split,
        runtime_cfg.task,
        runtime_cfg.context_length,
        runtime_cfg.prediction_length,
        runtime_cfg.max_windows,
    )
    batch_start = int(runtime_cfg.window_offset)
    batch_end = batch_start + int(runtime_cfg.batch_size)
    past = task_batch.past_values[batch_start:batch_end]
    future = None if task_batch.future_values is None else task_batch.future_values[batch_start:batch_end]
    target_values = None if task_batch.target_values is None else task_batch.target_values[batch_start:batch_end]
    if int(past.shape[0]) < runtime_cfg.batch_size:
        raise ValueError(
            "Dataset split does not contain enough task samples for requested batch size. "
            f"samples={task_batch.past_values.shape[0]}, batch_size={runtime_cfg.batch_size}, window_offset={batch_start}"
        )
    input_channels = int(past.shape[-1])
    checkpoint_channels = int(reference.config.num_input_channels)
    if input_channels != checkpoint_channels:
        if reference.task in {"forecast", "pretraining"} and runtime_cfg.allow_reference_channel_adaptation:
            adapt_reference_for_runtime_channels(reference, input_channels)
            checkpoint_channels = int(reference.config.num_input_channels)
        elif reference.task in {"forecast", "pretraining"}:
            raise ValueError(
                "Runtime channel count differs from the checkpoint channel count, but this run requires a native channel-shaped checkpoint. "
                f"input_channels={input_channels}, checkpoint_channels={checkpoint_channels}, checkpoint_id={reference.checkpoint_id}"
            )
    assert input_channels == checkpoint_channels, (
        "Input channel count does not match checkpoint channel count. "
        f"input_channels={input_channels}, checkpoint_channels={checkpoint_channels}"
    )
    if future is not None:
        assert int(future.shape[-1]) == checkpoint_channels, (
            "Input channel count does not match checkpoint channel count. "
            f"input_channels={int(future.shape[-1])}, checkpoint_channels={checkpoint_channels}"
        )
    if runtime_cfg.task == "forecast" and future is None:
        raise ValueError("Forecast task requires future_values from the real forecasting dataset.")
    if runtime_cfg.task in {"regression", "classification"} and target_values is None:
        raise ValueError(f"{runtime_cfg.task.title()} task requires labeled targets from the dataset adapter.")
    if runtime_cfg.task == "multi_task" and (future is None or target_values is None):
        raise ValueError("Multi-task forecast+classification requires both future_values and class targets.")
    observed = build_observed_mask(past)
    runtime_policy = resolve_runtime_policy_for_workload(runtime_policy, input_channels)

    # The demo owns only the runtime path: load data, run the model once, return the prediction.
    ttnn.CONFIG.throw_exception_on_fallback = True
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=DEFAULT_L1_SMALL_SIZE,
        trace_region_size=DEFAULT_TRACE_REGION_SIZE,
        num_command_queues=DEFAULT_NUM_COMMAND_QUEUES,
    )
    try:
        model = PatchTSTTTNNModel(
            demo_config=runtime_cfg,
            reference=reference,
            classification_reference=classification_reference,
            device=device,
            runtime_policy=runtime_policy,
        )
        try:
            if runtime_cfg.use_trace:
                session = _build_trace_session(model=model, past=past, observed=observed, task=runtime_cfg.task)
                try:
                    session.replay()
                    output = session.materialize()
                finally:
                    session.release()
            else:
                if runtime_cfg.task == "multi_task":
                    output = model.forward(past_values=past, past_observed_mask=observed, task="multi_task")
                else:
                    output = model.forward(
                        past_values=past,
                        past_observed_mask=observed,
                        task=runtime_cfg.task,
                    )
            prediction = output.prediction
            output.release()
            return prediction
        finally:
            model.close()
    finally:
        ttnn.close_device(device)


def run_streaming_forecast(
    config: PatchTSTDemoConfig,
    stream_steps: int,
    runtime_policy: PatchTSTRuntimePolicy | None = None,
    streaming_mode: Literal["cached", "full-rerun"] = "cached",
):
    _configure_host_torch_threads()
    reference = load_reference_model(
        "forecast",
        config.checkpoint_for_task("forecast"),
        config,
        config.checkpoint_revision_for_task("forecast"),
    )
    runtime_cfg = resolve_runtime_config(config, reference.config)
    matrix = load_dataset_matrix(
        dataset_root=runtime_cfg.dataset_root,
        dataset_name=runtime_cfg.dataset,
        split=runtime_cfg.split,
    )
    required_rows = runtime_cfg.context_length + stream_steps
    if matrix.shape[0] < required_rows:
        raise ValueError(
            "Dataset split does not contain enough rows for streaming run. "
            f"rows={matrix.shape[0]}, required={required_rows}"
        )

    context = matrix[: runtime_cfg.context_length].unsqueeze(0)
    stream_values = matrix[runtime_cfg.context_length : runtime_cfg.context_length + stream_steps].unsqueeze(0)
    if int(context.shape[-1]) != int(reference.config.num_input_channels):
        adapt_reference_for_runtime_channels(reference, runtime_num_channels=int(context.shape[-1]))
    expected_channels = int(reference.config.num_input_channels)
    assert int(context.shape[-1]) == expected_channels, (
        "Input channel count does not match checkpoint channel count. "
        f"input_channels={int(context.shape[-1])}, checkpoint_channels={expected_channels}"
    )
    assert int(stream_values.shape[-1]) == expected_channels, (
        "Input channel count does not match checkpoint channel count. "
        f"input_channels={int(stream_values.shape[-1])}, checkpoint_channels={expected_channels}"
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
            demo_config=runtime_cfg,
            reference=reference,
            device=device,
            runtime_policy=runtime_policy or PatchTSTRuntimePolicy(),
        )
        try:
            observed_mask = build_observed_mask(context)
            if streaming_mode == "cached":
                streamer = CachedForecastStreamer(
                    model=model,
                    initial_context=context,
                    initial_observed_mask=observed_mask,
                    use_trace=bool(runtime_cfg.use_trace),
                )
            else:
                streamer = HostFullRerunPatchTSTStreamer(
                    model=model,
                    initial_context=context,
                    initial_observed_mask=observed_mask,
                )
            try:
                predictions = []
                for step_idx in range(stream_steps):
                    tick = stream_values[:, step_idx : step_idx + 1, :]
                    predictions.append(streamer.step(tick))
                return predictions
            finally:
                close = getattr(streamer, "close", None)
                if callable(close):
                    close()
        finally:
            model.close()
    finally:
        ttnn.close_device(device)
