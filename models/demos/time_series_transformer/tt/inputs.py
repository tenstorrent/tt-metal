# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side construction of the transformer input sequence.

Scaling, lag gathering and covariate assembly are data preparation: they run once per forward
pass, cost nothing next to the decode loop, and mirror HuggingFace's ``create_network_inputs``
one-for-one, which keeps parity debugging tractable. Everything from the value embedding onward
runs on device.

The per-step lag gather during generation is the one piece of this that sits in the hot loop,
so it is handled separately: on device inside the traced rollout, and via :func:`get_latest_lags`
on the sampling path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import TimeSeriesTransformerConfig


@dataclass
class NetworkInputs:
    """Output of :func:`create_network_inputs`."""

    transformer_inputs: torch.Tensor  # (batch, context+prediction, feature_size)
    loc: torch.Tensor  # (batch, 1) location used to de-normalize predictions
    scale: torch.Tensor  # (batch, 1) scale used to de-normalize predictions
    static_feat: torch.Tensor  # (batch, num_static_features)


def mean_scaler(
    data: torch.Tensor,
    observed_indicator: torch.Tensor,
    *,
    minimum_scale: float,
    default_scale: Optional[float],
    dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``TimeSeriesMeanScaler``: scale by the mean absolute observed value."""
    ts_sum = (data * observed_indicator).abs().sum(dim, keepdim=True)
    num_observed = observed_indicator.sum(dim, keepdim=True)
    scale = ts_sum / torch.clamp(num_observed, min=1)

    if default_scale is None:
        # Series with no observations fall back to the batch-level average scale.
        batch_sum = ts_sum.sum(dim=0)
        batch_observations = torch.clamp(num_observed.sum(0), min=1)
        fallback = torch.squeeze(batch_sum / batch_observations)
    else:
        fallback = default_scale * torch.ones_like(scale)

    scale = torch.where(num_observed > 0, scale, fallback)
    scale = torch.clamp(scale, min=minimum_scale)
    return torch.zeros_like(scale), scale


def std_scaler(
    data: torch.Tensor,
    observed_indicator: torch.Tensor,
    *,
    minimum_scale: float,
    dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``TimeSeriesStdScaler``: standardize by observed mean and standard deviation."""
    denominator = observed_indicator.sum(dim, keepdim=True).clamp_min(1.0)
    loc = (data * observed_indicator).sum(dim, keepdim=True) / denominator
    variance = (((data - loc) * observed_indicator) ** 2).sum(dim, keepdim=True) / denominator
    return loc, (variance + minimum_scale).sqrt()


def apply_scaler(
    config: TimeSeriesTransformerConfig,
    data: torch.Tensor,
    observed_indicator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.scaling == "mean":
        return mean_scaler(
            data,
            observed_indicator,
            minimum_scale=config.mean_minimum_scale,
            default_scale=config.default_scale,
        )
    if config.scaling == "std":
        return std_scaler(data, observed_indicator, minimum_scale=config.std_minimum_scale)
    if config.scaling == "none":
        shape = data.shape[:1] + (1,) + data.shape[2:]
        return torch.zeros(shape, dtype=data.dtype), torch.ones(shape, dtype=data.dtype)
    raise ValueError(f"Unsupported scaling: {config.scaling}")


def embed_static_categorical(
    features: Optional[torch.Tensor],
    embedder_weights: list[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``TimeSeriesFeatureEmbedder``: one embedding table per categorical column."""
    if features is None or not embedder_weights:
        return None
    if features.dim() == 1:
        features = features.unsqueeze(-1)
    slices = torch.chunk(features, len(embedder_weights), dim=-1)
    embedded = [
        torch.nn.functional.embedding(part.squeeze(-1).long(), weight) for weight, part in zip(embedder_weights, slices)
    ]
    return torch.cat(embedded, dim=-1)


def get_latest_lags(sequence: torch.Tensor, lags_sequence: tuple[int, ...]) -> torch.Tensor:
    """Gather only the newest lag row -- the ``subsequences_length=1, shift=1`` case.

    Equivalent to ``get_lagged_subsequences(...)[:, -1:]`` but as a single indexed read
    instead of one slice per lag. This runs once per decode step, so the constant matters.
    """
    length = sequence.shape[1]
    columns = [length - lag for lag in lags_sequence]
    if min(columns) < 0:
        raise ValueError(f"Lag {max(lags_sequence)} exceeds history length {length}.")

    gathered = sequence[:, columns]
    if gathered.dim() == 2:
        return gathered.reshape(sequence.shape[0], 1, len(lags_sequence))
    # Multivariate: gathering gives (batch, lags, channels), but get_lagged_subsequences
    # stacks lags last and flattens channel-major, so transpose before flattening to keep the
    # same element order as the full gather.
    return gathered.transpose(1, 2).reshape(sequence.shape[0], 1, -1)


def get_lagged_subsequences(
    sequence: torch.Tensor,
    *,
    subsequences_length: int,
    lags_sequence: tuple[int, ...],
    shift: int = 0,
) -> torch.Tensor:
    """Stack ``len(lags_sequence)`` lagged windows along a new trailing axis."""
    sequence_length = sequence.shape[1]
    indices = [lag - shift for lag in lags_sequence]
    if max(indices) + subsequences_length > sequence_length:
        raise ValueError(
            f"Lag {max(indices)} plus subsequence length {subsequences_length} exceeds "
            f"history length {sequence_length}."
        )
    windows = []
    for lag_index in indices:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        windows.append(sequence[:, begin_index:end_index, ...])
    return torch.stack(windows, dim=-1)


def build_static_features(
    config: TimeSeriesTransformerConfig,
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    static_real_features: Optional[torch.Tensor],
    static_categorical_features: Optional[torch.Tensor],
    embedder_weights: list[torch.Tensor],
) -> torch.Tensor:
    """Assemble ``[embedded_categorical | static_real | log|loc| | log scale]``."""
    if config.input_size == 1:
        log_abs_loc = loc.abs().log1p()
        log_scale = scale.log()
    else:
        log_abs_loc = loc.squeeze(1).abs().log1p()
        log_scale = scale.squeeze(1).log()

    static_feat = torch.cat((log_abs_loc, log_scale), dim=1)
    if static_real_features is not None:
        static_feat = torch.cat((static_real_features, static_feat), dim=1)
    embedded = embed_static_categorical(static_categorical_features, embedder_weights)
    if embedded is not None:
        static_feat = torch.cat((embedded, static_feat), dim=1)
    return static_feat


def expand_features(static_feat: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
    """Broadcast static features across time and concatenate the time covariates."""
    expanded = static_feat.unsqueeze(1).expand(-1, time_features.shape[1], -1)
    return torch.cat((expanded, time_features), dim=-1)


def create_network_inputs(
    config: TimeSeriesTransformerConfig,
    *,
    past_values: torch.Tensor,
    past_time_features: torch.Tensor,
    past_observed_mask: Optional[torch.Tensor] = None,
    static_categorical_features: Optional[torch.Tensor] = None,
    static_real_features: Optional[torch.Tensor] = None,
    future_values: Optional[torch.Tensor] = None,
    future_time_features: Optional[torch.Tensor] = None,
    embedder_weights: Optional[list[torch.Tensor]] = None,
) -> NetworkInputs:
    """Mirror of ``TimeSeriesTransformerModel.create_network_inputs``."""
    embedder_weights = embedder_weights or []
    if past_observed_mask is None:
        past_observed_mask = torch.ones_like(past_values)

    context_start = config.past_length - config.context_length
    time_feat = past_time_features[:, context_start:, ...]
    if future_values is not None and future_time_features is not None:
        time_feat = torch.cat((time_feat, future_time_features), dim=1)

    context = past_values[:, -config.context_length :]
    observed_context = past_observed_mask[:, -config.context_length :]
    loc, scale = apply_scaler(config, context, observed_context)

    if future_values is not None:
        inputs = (torch.cat((past_values, future_values), dim=1) - loc) / scale
        subsequences_length = config.context_length + config.prediction_length
    else:
        inputs = (past_values - loc) / scale
        subsequences_length = config.context_length

    static_feat = build_static_features(
        config,
        loc=loc,
        scale=scale,
        static_real_features=static_real_features,
        static_categorical_features=static_categorical_features,
        embedder_weights=embedder_weights,
    )
    features = expand_features(static_feat, time_feat)

    lagged = get_lagged_subsequences(
        inputs,
        subsequences_length=subsequences_length,
        lags_sequence=config.lags_sequence,
    )
    reshaped_lagged = lagged.reshape(lagged.shape[0], lagged.shape[1], -1)
    if reshaped_lagged.shape[1] != features.shape[1]:
        raise ValueError(
            f"Lagged sequence length {reshaped_lagged.shape[1]} does not match "
            f"covariate length {features.shape[1]}."
        )

    return NetworkInputs(
        transformer_inputs=torch.cat((reshaped_lagged, features), dim=-1),
        loc=loc,
        scale=scale,
        static_feat=static_feat,
    )


__all__ = [
    "NetworkInputs",
    "apply_scaler",
    "build_static_features",
    "create_network_inputs",
    "embed_static_categorical",
    "expand_features",
    "get_lagged_subsequences",
    "get_latest_lags",
    "mean_scaler",
    "std_scaler",
]
