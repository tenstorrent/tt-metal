# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import namedtuple

import torch

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, resolve_runtime_config
from models.demos.wormhole.patchtst.demo.data_utils import build_observed_mask, load_task_dataset
from models.demos.wormhole.patchtst.reference.hf_reference import (
    adapt_reference_for_runtime_channels,
    adapt_reference_for_runtime_context,
    load_reference_model,
)

PreparedRun = namedtuple(
    "PreparedRun",
    ["reference", "classification_reference", "runtime_cfg", "past", "future", "target_values", "observed"],
)


def compute_metrics(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    prediction = prediction.to(torch.float32)
    reference = reference.to(torch.float32)
    if not torch.isfinite(prediction).all():
        raise ValueError("Prediction tensor contains NaN or Inf values.")
    if not torch.isfinite(reference).all():
        raise ValueError("Reference tensor contains NaN or Inf values.")
    diff = prediction - reference
    pred_flat = prediction.reshape(-1)
    ref_flat = reference.reshape(-1)
    fallback_corr = torch.clamp(
        1.0 - (torch.mean(torch.abs(pred_flat - ref_flat)) / torch.clamp(torch.mean(torch.abs(ref_flat)), min=1.0)),
        min=0.0,
        max=1.0,
    )
    if pred_flat.numel() < 2:
        corr = fallback_corr
    else:
        pred_centered = pred_flat - pred_flat.mean()
        ref_centered = ref_flat - ref_flat.mean()
        denom = torch.sqrt(torch.sum(pred_centered * pred_centered) * torch.sum(ref_centered * ref_centered))
        corr = (
            fallback_corr
            if float(denom.item()) < 1e-12
            else torch.clamp(torch.sum(pred_centered * ref_centered) / denom, min=-1.0, max=1.0)
        )
    return {
        "mse": float(torch.mean(diff * diff).item()),
        "mae": float(torch.mean(torch.abs(diff)).item()),
        "correlation": float(corr.item()),
    }


def compute_classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target = target.to(torch.long).reshape(-1)
    prediction = torch.argmax(logits.to(torch.float32), dim=-1).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(
            "Classification logits and targets must produce the same flattened shape. "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
    accuracy = float((prediction == target).to(torch.float32).mean().item())
    max_class = int(max(prediction.max().item(), target.max().item())) if prediction.numel() > 0 else -1
    f1_scores = []
    for class_idx in range(max_class + 1):
        pred_positive = prediction == class_idx
        target_positive = target == class_idx
        tp = int((pred_positive & target_positive).sum().item())
        fp = int((pred_positive & ~target_positive).sum().item())
        fn = int((~pred_positive & target_positive).sum().item())
        denom = (2 * tp) + fp + fn
        f1_scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return {"accuracy": accuracy, "f1_macro": float(sum(f1_scores) / len(f1_scores) if f1_scores else 0.0)}


def prepare_run(config: PatchTSTDemoConfig) -> PreparedRun:
    reference = load_reference_model(
        "forecast" if config.task == "multi_task" else config.task,
        config.multi_task_checkpoint_ids["forecast"]
        if config.task == "multi_task"
        else config.effective_checkpoint_id(),
        config,
        config.multi_task_checkpoint_revisions["forecast"]
        if config.task == "multi_task"
        else config.effective_checkpoint_revision(),
    )
    classification_reference = (
        load_reference_model(
            "classification",
            config.multi_task_checkpoint_ids["classification"],
            config,
            config.multi_task_checkpoint_revisions["classification"],
        )
        if config.task == "multi_task"
        else None
    )

    runtime_cfg = resolve_runtime_config(config, reference.config)
    if int(reference.config.context_length) != int(runtime_cfg.context_length):
        if not runtime_cfg.allow_reference_context_adaptation:
            raise ValueError(
                "Runtime context differs from the checkpoint context, but this run requires a native context-shaped checkpoint. "
                f"runtime_context={runtime_cfg.context_length}, checkpoint_context={reference.config.context_length}, "
                f"checkpoint_id={reference.checkpoint_id}"
            )
        adapt_reference_for_runtime_context(reference, runtime_context_length=runtime_cfg.context_length)

    task_batch = load_task_dataset(
        dataset_root=runtime_cfg.dataset_root,
        dataset_name=runtime_cfg.dataset,
        split=runtime_cfg.split,
        task=runtime_cfg.task,
        context_length=runtime_cfg.context_length,
        prediction_length=runtime_cfg.prediction_length,
        max_windows=runtime_cfg.max_windows,
    )
    batch_start = int(runtime_cfg.window_offset)
    batch_end = batch_start + int(runtime_cfg.batch_size)
    past = task_batch.past_values[batch_start:batch_end]
    future = task_batch.future_values[batch_start:batch_end] if task_batch.future_values is not None else None
    target_values = task_batch.target_values[batch_start:batch_end] if task_batch.target_values is not None else None
    if int(past.shape[0]) < runtime_cfg.batch_size:
        raise ValueError(
            "Dataset split does not contain enough task samples for requested batch size. "
            f"samples={task_batch.past_values.shape[0]}, batch_size={runtime_cfg.batch_size}, window_offset={batch_start}"
        )

    dataset_channels = int(past.shape[-1])
    expected_channels = int(reference.config.num_input_channels)
    if dataset_channels != expected_channels:
        if reference.task in {"forecast", "pretraining"} and runtime_cfg.allow_reference_channel_adaptation:
            adapt_reference_for_runtime_channels(reference, runtime_num_channels=dataset_channels)
            expected_channels = int(reference.config.num_input_channels)
        elif reference.task in {"forecast", "pretraining"}:
            raise ValueError(
                "Runtime channel count differs from the checkpoint channel count, but this run requires a native channel-shaped checkpoint. "
                f"input_channels={dataset_channels}, checkpoint_channels={expected_channels}, checkpoint_id={reference.checkpoint_id}"
            )

    assert int(past.shape[-1]) == expected_channels, (
        "Input channel count does not match checkpoint channel count. "
        f"input_channels={int(past.shape[-1])}, checkpoint_channels={expected_channels}"
    )
    if future is not None:
        assert int(future.shape[-1]) == expected_channels, (
            "Input channel count does not match checkpoint channel count. "
            f"input_channels={int(future.shape[-1])}, checkpoint_channels={expected_channels}"
        )
    if runtime_cfg.task == "forecast" and future is None:
        raise ValueError("Forecast task requires future_values from the real forecasting dataset.")
    if runtime_cfg.task in {"regression", "classification"} and target_values is None:
        raise ValueError(f"{runtime_cfg.task.title()} task requires labeled targets from the dataset adapter.")
    if runtime_cfg.task == "multi_task" and (future is None or target_values is None):
        raise ValueError("Multi-task forecast+classification requires both future_values and class targets.")
    return PreparedRun(
        reference, classification_reference, runtime_cfg, past, future, target_values, build_observed_mask(past)
    )
