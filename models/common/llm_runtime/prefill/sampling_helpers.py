# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stateless prefill sampling helpers."""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

import ttnn
from models.common.llm_runtime.prefill.plan import _round_up
from models.common.sampling import format_sampling_params

_TILE_SIZE = 32

SamplingPath = Literal["logits", "argmax", "topk"]


def _slice_sampling_params(sampling_params, source_rows):
    if sampling_params is None:
        return None
    if not dataclasses.is_dataclass(sampling_params):
        raise TypeError("sampling_params must be a dataclass")

    def slice_value(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            selected_rows = (0,) * len(source_rows) if int(value.shape[0]) == 1 else source_rows
            indices = torch.tensor(selected_rows, dtype=torch.long, device=value.device)
            return value.index_select(0, indices)
        if isinstance(value, list):
            return [value[0] for _ in source_rows] if len(value) == 1 else [value[row] for row in source_rows]
        if isinstance(value, tuple):
            return tuple(value[0] for _ in source_rows) if len(value) == 1 else tuple(value[row] for row in source_rows)
        return value

    updates = {
        field.name: slice_value(getattr(sampling_params, field.name)) for field in dataclasses.fields(sampling_params)
    }
    return dataclasses.replace(sampling_params, **updates)


def _formatted_sampling_values(sampling_params, batch_size):
    sampling_params = _tensor_sampling_fields_to_python(sampling_params)
    formatted_size = _round_up(int(batch_size), _TILE_SIZE)
    formatted = format_sampling_params(sampling_params, formatted_size)
    k = tuple(int(value) for value in formatted.top_k[:batch_size])
    p = tuple(float(value) for value in formatted.top_p[:batch_size])
    temperature = tuple(float(value) for value in formatted.temperature[:batch_size])
    greedy = (
        all(value == 1 for value in k) and all(value == 0 for value in p) and all(value == 1 for value in temperature)
    )
    return k, p, temperature, greedy


def _tensor_sampling_fields_to_python(sampling_params):
    updates = {}
    for field in dataclasses.fields(sampling_params):
        value = getattr(sampling_params, field.name)
        if isinstance(value, torch.Tensor):
            updates[field.name] = value.item() if value.ndim == 0 else value.tolist()
    return dataclasses.replace(sampling_params, **updates) if updates else sampling_params


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
    if len(row_payloads) == 1 and row_payloads[0][0] == tuple(range(batch_size)):
        return row_payloads[0][1]
    ordered = [None] * batch_size
    for rows, payload in row_payloads:
        if isinstance(payload, torch.Tensor) and payload.shape[0] == len(rows):
            for local_row, source_row in enumerate(rows):
                ordered[source_row] = payload[local_row]
        else:
            for source_row in rows:
                ordered[source_row] = payload
    return ordered
