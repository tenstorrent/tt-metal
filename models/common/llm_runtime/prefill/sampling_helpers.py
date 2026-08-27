# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stateless prefill sampling helpers."""

from __future__ import annotations

from typing import Literal

import torch

import ttnn
from models.common.modules.sampling.params import prepare_sampling_params, slice_sampling_params

_TILE_SIZE = 32

SamplingPath = Literal["logits", "argmax", "topk"]


def _slice_sampling_params(sampling_params, source_rows):
    if sampling_params is None:
        return None
    return slice_sampling_params(sampling_params, source_rows)


def _formatted_sampling_values(
    sampling_params,
    batch_size,
    *,
    max_device_top_k=32,
    allow_force_argmax=True,
):
    """Compatibility test helper backed by the native exact formatter."""

    prepared = prepare_sampling_params(
        sampling_params,
        batch_size,
        max_device_top_k=max_device_top_k,
        allow_force_argmax=allow_force_argmax,
    )
    return prepared.top_k, prepared.top_p, prepared.temperature, prepared.all_active_rows_greedy


def _select_sample_log_prob(value, row):
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[int(row)]
    if isinstance(value, ttnn.Tensor):
        first_replica = ttnn.get_device_tensors(value)[0]
        return ttnn.to_torch(first_replica).reshape(-1)[int(row)]
    return value


def _merge_log_probs(row_payloads, batch_size):
    if not row_payloads:
        return None
    ordered = torch.ones(int(batch_size), dtype=torch.float32)
    for rows, payload in row_payloads:
        values = _sampled_log_probs_for_rows(payload, len(rows))
        indices = torch.tensor(tuple(int(row) for row in rows), dtype=torch.long)
        ordered.index_copy_(0, indices, values)
    return ordered


def _sampled_log_probs_for_rows(value, row_count):
    """Flatten Sampling1D's replicated sampled-token logprob output."""

    if isinstance(value, torch.Tensor):
        output = value
    elif isinstance(value, ttnn.Tensor):
        replicas = ttnn.get_device_tensors(value)
        output = ttnn.to_torch(replicas[0] if replicas else value)
    elif isinstance(value, (float, int)):
        return torch.full((int(row_count),), float(value), dtype=torch.float32)
    elif isinstance(value, (list, tuple)):
        output = torch.as_tensor(value)
    else:
        raise TypeError(
            "sampled-token logprobs must be a TT tensor, Torch tensor, or numeric sequence"
        )
    flat = output.reshape(-1)
    if int(flat.numel()) == 1 and int(row_count) > 1:
        flat = flat.expand(int(row_count))
    if int(flat.numel()) < int(row_count):
        raise ValueError(
            f"sampled-token logprobs contain {flat.numel()} rows, expected at least {row_count}"
        )
    return flat[: int(row_count)].to(torch.float32)
