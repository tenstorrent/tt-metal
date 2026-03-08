# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Any

import torch


def standardize_per_channel(history: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = history.mean(dim=1, keepdim=True)
    std = history.std(dim=1, keepdim=True).clamp_min(eps)
    normalized = (history - mean) / std
    return normalized, mean, std


def sliding_windows(
    series: torch.Tensor,
    *,
    context_length: int,
    forecast_length: int,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if series.dim() != 2:
        raise ValueError(f"Expected [time, channels] series, got {tuple(series.shape)}")

    total_length = context_length + forecast_length
    if series.shape[0] < total_length:
        raise ValueError(f"Need at least {total_length} timesteps, got {series.shape[0]}")

    histories = []
    targets = []
    for start in range(0, series.shape[0] - total_length + 1, stride):
        mid = start + context_length
        end = mid + forecast_length
        histories.append(series[start:mid])
        targets.append(series[mid:end])

    return torch.stack(histories, dim=0), torch.stack(targets, dim=0)


def build_reference_inputs(
    model: torch.nn.Module,
    history: torch.Tensor,
    *,
    future_values: torch.Tensor | None = None,
    observed_mask: torch.Tensor | None = None,
    extra_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    param_names = set(inspect.signature(model.forward).parameters.keys())
    inputs: dict[str, Any] = {}

    if "past_values" in param_names:
        inputs["past_values"] = history
    elif "x_enc" in param_names:
        inputs["x_enc"] = history
    elif "input_ids" in param_names:
        inputs["input_ids"] = history
    else:
        first_parameter = next(iter(param_names), None)
        if first_parameter is not None:
            inputs[first_parameter] = history

    if observed_mask is None:
        observed_mask = torch.ones_like(history, dtype=history.dtype)

    if "past_observed_mask" in param_names:
        inputs["past_observed_mask"] = observed_mask
    if "observed_mask" in param_names:
        inputs["observed_mask"] = observed_mask

    if future_values is not None:
        if "future_values" in param_names:
            inputs["future_values"] = future_values
        if "labels" in param_names:
            inputs["labels"] = future_values
        if "x_dec" in param_names:
            inputs["x_dec"] = future_values
        if "future_observed_mask" in param_names:
            inputs["future_observed_mask"] = torch.ones_like(future_values, dtype=future_values.dtype)

    if extra_inputs:
        for key, value in extra_inputs.items():
            if key in param_names:
                inputs[key] = value

    return inputs
