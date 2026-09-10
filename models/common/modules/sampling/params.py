# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Topology-neutral preparation of TTTv2 sampling request parameters.

This module deliberately contains no mesh or TTNN policy.  A caller resolves
the sampler capabilities from its ``Sampling1DConfig`` and passes them here.
The resulting immutable value contains every request-owned sampling field in
slot order and is safe to slice or retain across eager and trace lifecycles.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import torch

from models.common.sampling.sampling_params import SamplingParams

SamplingPath = Literal["argmax", "topk"]
SamplingRowPath = Literal["inactive", "argmax", "topk"]
LogProbMode = Literal["none", "sampled_token", "top_n"]


@dataclass(frozen=True)
class PreparedSamplingParams:
    """One normalized device-sampling request.

    All row-owned tuples have exactly ``batch_size`` entries.  ``temperature``
    holds the inverse temperature consumed by ``Sampling1D``.  Greedy rows use
    the device representation ``top_k=1``, ``top_p=0`` and ``temperature=1``.

    ``sampling_path`` is batch-wide because the common runtime selects one
    program for the entire lane.  A mixed greedy/stochastic request therefore
    uses ``topk`` even when force-argmax is available.
    """

    top_k: tuple[int, ...]
    top_p: tuple[float, ...]
    temperature: tuple[float, ...]
    presence_penalty: tuple[float, ...]
    frequency_penalty: tuple[float, ...]
    repetition_penalty: tuple[float, ...]
    seeds: tuple[int | None, ...]
    enable_log_probs: tuple[bool, ...]
    num_logprobs: tuple[int, ...]
    logprob_modes: tuple[LogProbMode, ...]
    greedy_mask: tuple[bool, ...]
    row_paths: tuple[SamplingRowPath, ...]
    active_mask: tuple[bool, ...]
    sampling_path: SamplingPath
    active_rows: int
    batch_size: int
    max_device_top_k: int
    prompt_tokens: Any | None = None
    output_tokens: Any | None = None
    slot_remap: Any | None = None

    def __post_init__(self) -> None:
        if self.active_rows <= 0 or self.active_rows > self.batch_size:
            raise ValueError("active_rows must be in [1, batch_size]")
        row_fields = (
            "top_k",
            "top_p",
            "temperature",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seeds",
            "enable_log_probs",
            "num_logprobs",
            "logprob_modes",
            "greedy_mask",
            "row_paths",
            "active_mask",
        )
        for name in row_fields:
            if len(getattr(self, name)) != self.batch_size:
                raise ValueError(f"{name} must contain exactly batch_size entries")
        if sum(self.active_mask) != self.active_rows:
            raise ValueError("active_rows must equal the number of active_mask entries")

    @property
    def penalties_enabled(self) -> bool:
        return (
            any(active and value != 0.0 for active, value in zip(self.active_mask, self.presence_penalty))
            or any(active and value != 0.0 for active, value in zip(self.active_mask, self.frequency_penalty))
            or any(active and value != 1.0 for active, value in zip(self.active_mask, self.repetition_penalty))
        )

    @property
    def log_probs_enabled(self) -> bool:
        return any(active and mode != "none" for active, mode in zip(self.active_mask, self.logprob_modes))

    @property
    def all_active_rows_greedy(self) -> bool:
        return all(greedy for active, greedy in zip(self.active_mask, self.greedy_mask) if active)

    @property
    def all_active_rows_argmax(self) -> bool:
        return all(path == "argmax" for active, path in zip(self.active_mask, self.row_paths) if active)


_DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": None,
    "enable_log_probs": False,
    "num_logprobs": 0,
}


def prepare_sampling_params(
    sampling_params: SamplingParams,
    batch_size: int,
    *,
    max_device_top_k: int,
    allow_force_argmax: bool,
    prompt_tokens: Any | None = None,
    output_tokens: Any | None = None,
    slot_remap: Any | None = None,
) -> PreparedSamplingParams:
    """Normalize, validate, and classify a TTTv2 sampling request.

    Unsupported stochastic ``top_k`` values are rejected rather than clamped.
    The check intentionally happens after greedy rows are normalized, because
    vLLM commonly represents unrestricted ``top_k`` as the vocabulary size even
    for a request whose temperature is zero.
    """

    _validate_policy(batch_size, max_device_top_k, allow_force_argmax)
    _validate_sampling_value(sampling_params)

    temperature_input = _as_sequence(getattr(sampling_params, "temperature"), "temperature")
    active_rows = len(temperature_input)
    if active_rows > batch_size:
        raise ValueError(f"temperature describes {active_rows} active rows, exceeding batch_size={batch_size}")

    temperature = _normalize_per_row(temperature_input, "temperature", active_rows, batch_size)
    top_p = _normalize_per_row(getattr(sampling_params, "top_p"), "top_p", active_rows, batch_size)
    top_k = _normalize_per_row(getattr(sampling_params, "top_k"), "top_k", active_rows, batch_size)
    presence_penalty = _normalize_per_row(
        getattr(sampling_params, "presence_penalty", _DEFAULTS["presence_penalty"]),
        "presence_penalty",
        active_rows,
        batch_size,
    )
    frequency_penalty = _normalize_per_row(
        getattr(sampling_params, "frequency_penalty", _DEFAULTS["frequency_penalty"]),
        "frequency_penalty",
        active_rows,
        batch_size,
    )
    repetition_penalty = _normalize_per_row(
        getattr(sampling_params, "repetition_penalty", _DEFAULTS["repetition_penalty"]),
        "repetition_penalty",
        active_rows,
        batch_size,
    )
    seeds = _normalize_seeds(getattr(sampling_params, "seed", None), batch_size)
    enable_log_probs = _normalize_output_field(
        getattr(sampling_params, "enable_log_probs", False),
        "enable_log_probs",
        batch_size,
    )
    num_logprobs_value = getattr(sampling_params, "num_logprobs", 0)
    num_logprobs = _normalize_output_field(
        0 if num_logprobs_value is None else num_logprobs_value,
        "num_logprobs",
        batch_size,
    )

    row_paths: list[SamplingRowPath] = ["inactive"] * batch_size
    active_mask = [row < active_rows for row in range(batch_size)]
    greedy_mask = [False] * batch_size
    logprob_modes: list[LogProbMode] = ["none"] * batch_size
    for row in range(batch_size):
        top_p[row] = min(max(float(top_p[row]), 0.0), 1.0)
        repetition_penalty[row] = float(repetition_penalty[row]) or 1.0

        if row >= active_rows:
            temperature[row] = 1.0
            top_k[row] = 1
            top_p[row] = 0.0
            enable_log_probs[row] = False
            num_logprobs[row] = 0
            continue

        is_greedy = float(temperature[row]) == 0.0
        greedy_mask[row] = is_greedy
        if is_greedy:
            temperature[row] = 1.0
            top_k[row] = 1
            top_p[row] = 0.0
        else:
            temperature[row] = 1.0 / float(temperature[row])
            top_k[row] = _exact_top_k(top_k[row], row=row, max_device_top_k=max_device_top_k)

        enabled = bool(enable_log_probs[row])
        count = int(num_logprobs[row])
        if not enabled:
            count = 0
            num_logprobs[row] = 0
        elif count < 0:
            raise ValueError(f"sampling_params.num_logprobs[{row}] must be non-negative, got {count}")
        logprob_modes[row] = "none" if not enabled else ("sampled_token" if count == 0 else "top_n")
        row_paths[row] = "argmax" if is_greedy and allow_force_argmax and not enabled else "topk"

    sampling_path: SamplingPath = "argmax" if all(path == "argmax" for path in row_paths[:active_rows]) else "topk"
    return PreparedSamplingParams(
        top_k=tuple(int(value) for value in top_k),
        top_p=tuple(float(value) for value in top_p),
        temperature=tuple(float(value) for value in temperature),
        presence_penalty=tuple(float(value) for value in presence_penalty),
        frequency_penalty=tuple(float(value) for value in frequency_penalty),
        repetition_penalty=tuple(float(value) for value in repetition_penalty),
        seeds=tuple(None if value is None else int(value) for value in seeds),
        enable_log_probs=tuple(bool(value) for value in enable_log_probs),
        num_logprobs=tuple(int(value) for value in num_logprobs),
        logprob_modes=tuple(logprob_modes),
        greedy_mask=tuple(greedy_mask),
        row_paths=tuple(row_paths),
        active_mask=tuple(active_mask),
        sampling_path=sampling_path,
        active_rows=active_rows,
        batch_size=batch_size,
        max_device_top_k=max_device_top_k,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        slot_remap=slot_remap,
    )


def format_sampling_params(
    sampling_params: SamplingParams,
    batch_size: int,
    *,
    max_device_top_k: int,
    allow_force_argmax: bool,
    prompt_tokens: Any | None = None,
    output_tokens: Any | None = None,
    slot_remap: Any | None = None,
) -> PreparedSamplingParams:
    """Compatibility spelling for callers that describe this step as formatting."""

    return prepare_sampling_params(
        sampling_params,
        batch_size,
        max_device_top_k=max_device_top_k,
        allow_force_argmax=allow_force_argmax,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        slot_remap=slot_remap,
    )


def slice_prepared_sampling_params(
    prepared: PreparedSamplingParams,
    rows: Sequence[int],
) -> PreparedSamplingParams:
    """Slice a complete prepared request, including sampling-owned history.

    State tensors and sequences are indexed on their leading request dimension.
    Values with one leading row broadcast to the selected rows.  Slot-remap
    values themselves are preserved; this function only selects which request
    rows are assigned to the destination lane.
    """

    if not isinstance(prepared, PreparedSamplingParams):
        raise TypeError("prepared must be PreparedSamplingParams")
    selected = tuple(int(row) for row in rows)
    if not selected:
        raise ValueError("prepared sampling rows cannot be empty")
    if any(row < 0 or row >= prepared.batch_size for row in selected):
        raise ValueError(f"prepared sampling rows must be in [0, {prepared.batch_size})")

    active_mask = tuple(prepared.active_mask[row] for row in selected)
    active_rows = sum(active_mask)
    if active_rows == 0:
        raise ValueError("prepared sampling slice must include at least one active row")
    row_paths = tuple(prepared.row_paths[row] for row in selected)
    active_paths = tuple(path for active, path in zip(active_mask, row_paths) if active)
    sampling_path: SamplingPath = "argmax" if all(path == "argmax" for path in active_paths) else "topk"

    def select_tuple(value: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(value[row] for row in selected)

    return PreparedSamplingParams(
        top_k=select_tuple(prepared.top_k),
        top_p=select_tuple(prepared.top_p),
        temperature=select_tuple(prepared.temperature),
        presence_penalty=select_tuple(prepared.presence_penalty),
        frequency_penalty=select_tuple(prepared.frequency_penalty),
        repetition_penalty=select_tuple(prepared.repetition_penalty),
        seeds=select_tuple(prepared.seeds),
        enable_log_probs=select_tuple(prepared.enable_log_probs),
        num_logprobs=select_tuple(prepared.num_logprobs),
        logprob_modes=select_tuple(prepared.logprob_modes),
        greedy_mask=select_tuple(prepared.greedy_mask),
        row_paths=row_paths,
        active_mask=active_mask,
        sampling_path=sampling_path,
        active_rows=active_rows,
        batch_size=len(selected),
        max_device_top_k=prepared.max_device_top_k,
        prompt_tokens=_slice_request_state(prepared.prompt_tokens, selected, "prompt_tokens"),
        output_tokens=_slice_request_state(prepared.output_tokens, selected, "output_tokens"),
        slot_remap=_slice_request_state(prepared.slot_remap, selected, "slot_remap"),
    )


def place_prepared_sampling_params(
    prepared: PreparedSamplingParams,
    slots: Sequence[int],
) -> PreparedSamplingParams:
    """Place request-ordered active rows into lane-local destination slots.

    Prefill parameters arrive in request order while device K/P/T, seed, and
    penalty state are slot indexed.  This conversion preserves inactive safe
    defaults and expands prompt/output history to the fixed lane capacity.
    """

    if not isinstance(prepared, PreparedSamplingParams):
        raise TypeError("prepared must be PreparedSamplingParams")
    sources = tuple(row for row, active in enumerate(prepared.active_mask) if active)
    destinations = tuple(int(slot) for slot in slots)
    if len(destinations) != len(sources):
        raise ValueError(f"expected {len(sources)} destination slots, got {len(destinations)}")
    if len(set(destinations)) != len(destinations):
        raise ValueError("destination slots must be unique")
    if any(slot < 0 or slot >= prepared.batch_size for slot in destinations):
        raise ValueError(f"destination slots must be in [0, {prepared.batch_size})")

    def place(values: tuple[Any, ...], default: Any) -> tuple[Any, ...]:
        result = [default] * prepared.batch_size
        for source, destination in zip(sources, destinations):
            result[destination] = values[source]
        return tuple(result)

    row_paths = place(prepared.row_paths, "inactive")
    active_mask = tuple(path != "inactive" for path in row_paths)
    active_paths = tuple(path for path in row_paths if path != "inactive")
    sampling_path: SamplingPath = "argmax" if all(path == "argmax" for path in active_paths) else "topk"
    return PreparedSamplingParams(
        top_k=place(prepared.top_k, 1),
        top_p=place(prepared.top_p, 0.0),
        temperature=place(prepared.temperature, 1.0),
        presence_penalty=place(prepared.presence_penalty, 0.0),
        frequency_penalty=place(prepared.frequency_penalty, 0.0),
        repetition_penalty=place(prepared.repetition_penalty, 1.0),
        seeds=place(prepared.seeds, None),
        enable_log_probs=place(prepared.enable_log_probs, False),
        num_logprobs=place(prepared.num_logprobs, 0),
        logprob_modes=place(prepared.logprob_modes, "none"),
        greedy_mask=place(prepared.greedy_mask, False),
        row_paths=row_paths,
        active_mask=active_mask,
        sampling_path=sampling_path,
        active_rows=prepared.active_rows,
        batch_size=prepared.batch_size,
        max_device_top_k=prepared.max_device_top_k,
        prompt_tokens=_place_request_state(
            prepared.prompt_tokens,
            sources=sources,
            destinations=destinations,
            capacity=prepared.batch_size,
            name="prompt_tokens",
        ),
        output_tokens=_place_request_state(
            prepared.output_tokens,
            sources=sources,
            destinations=destinations,
            capacity=prepared.batch_size,
            name="output_tokens",
        ),
        slot_remap=prepared.slot_remap,
    )


def slice_sampling_params(sampling_params: SamplingParams, rows: Sequence[int]) -> SamplingParams:
    """Return request parameters for ``rows`` without mutating the caller value."""

    _validate_sampling_value(sampling_params)
    selected = tuple(int(row) for row in rows)
    if not selected:
        raise ValueError("sampling parameter rows cannot be empty")
    if any(row < 0 for row in selected):
        raise ValueError("sampling parameter rows must be non-negative")

    def slice_value(value: Any, name: str) -> Any:
        normalized = _host_value(value)
        if not _is_sequence(normalized):
            return normalized
        values = list(normalized)
        if not values:
            raise ValueError(f"sampling_params.{name} cannot be empty")
        if len(values) == 1:
            return [values[0] for _ in selected]
        try:
            return [values[row] for row in selected]
        except IndexError as error:
            raise ValueError(f"sampling_params.{name} does not cover rows {selected}") from error

    updates = {
        field.name: slice_value(getattr(sampling_params, field.name), field.name)
        for field in dataclasses.fields(sampling_params)
    }
    return dataclasses.replace(sampling_params, **updates)


def _validate_policy(batch_size: int, max_device_top_k: int, allow_force_argmax: bool) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(max_device_top_k, int) or isinstance(max_device_top_k, bool) or max_device_top_k <= 0:
        raise ValueError("max_device_top_k must be a positive integer")
    if not isinstance(allow_force_argmax, bool):
        raise TypeError("allow_force_argmax must be bool")


def _validate_sampling_value(sampling_params: Any) -> None:
    if not dataclasses.is_dataclass(sampling_params) or isinstance(sampling_params, type):
        raise TypeError("sampling_params must be a dataclass instance")
    for name in ("temperature", "top_k", "top_p"):
        if not hasattr(sampling_params, name):
            raise TypeError(f"sampling_params must define {name}")


def _host_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.reshape(-1).tolist()
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_sequence(value: Any, name: str) -> list[Any]:
    value = _host_value(value)
    values = list(value) if _is_sequence(value) else [value]
    if not values:
        raise ValueError(f"sampling_params.{name} cannot be empty")
    return values


def _normalize_per_row(value: Any, name: str, active_rows: int, batch_size: int) -> list[Any]:
    value = _host_value(value)
    if not _is_sequence(value):
        values = [value] * active_rows
    else:
        values = list(value)
        if not values:
            raise ValueError(f"sampling_params.{name} cannot be empty")
        if len(values) != 1 and len(values) < active_rows:
            raise ValueError(
                f"sampling_params.{name} has {len(values)} entries but temperature describes "
                f"{active_rows} active rows"
            )
    if len(values) > batch_size:
        raise ValueError(f"sampling_params.{name} has {len(values)} entries, exceeding batch_size={batch_size}")
    return values + [_DEFAULTS[name]] * (batch_size - len(values))


def _normalize_seeds(value: Any, batch_size: int) -> list[int | None]:
    value = _host_value(value)
    if value is None:
        values: list[int | None] = []
    elif _is_sequence(value):
        values = list(value)
    else:
        # Seed is request-owned and is never implicitly broadcast to sibling rows.
        values = [value]
    if len(values) > batch_size:
        raise ValueError(f"sampling_params.seed has {len(values)} entries, exceeding batch_size={batch_size}")
    normalized = [None if item is None or int(item) == -1 else int(item) for item in values]
    return normalized + [None] * (batch_size - len(normalized))


def _normalize_output_field(value: Any, name: str, batch_size: int) -> list[Any]:
    value = _host_value(value)
    if not _is_sequence(value):
        return [value] * batch_size
    values = list(value)
    if not values:
        raise ValueError(f"sampling_params.{name} cannot be empty")
    if len(values) == 1:
        return values * batch_size
    if len(values) > batch_size:
        raise ValueError(f"sampling_params.{name} has {len(values)} entries, exceeding batch_size={batch_size}")
    return values + [_DEFAULTS[name]] * (batch_size - len(values))


def _slice_request_state(value: Any, rows: tuple[int, ...], name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        selected = (0,) * len(rows) if int(value.shape[0]) == 1 else rows
        if max(selected) >= int(value.shape[0]):
            raise ValueError(f"{name} does not cover rows {rows}")
        indices = torch.tensor(selected, dtype=torch.long, device=value.device)
        return value.index_select(0, indices)
    if _is_sequence(value):
        values = list(value)
        if not values:
            raise ValueError(f"{name} cannot be empty")
        selected = (0,) * len(rows) if len(values) == 1 else rows
        try:
            sliced = [values[row] for row in selected]
        except IndexError as error:
            raise ValueError(f"{name} does not cover rows {rows}") from error
        return tuple(sliced) if isinstance(value, tuple) else sliced
    raise TypeError(f"{name} must be a row-indexed tensor or sequence")


def _place_request_state(
    value: Any,
    *,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    capacity: int,
    name: str,
) -> Any:
    if value is None:
        return None
    selected = _slice_request_state(value, sources, name)
    if isinstance(selected, torch.Tensor):
        if selected.ndim == 0:
            selected = selected.reshape(1)
        fill_value = False if selected.dtype == torch.bool else -1
        placed = torch.full(
            (capacity, *selected.shape[1:]),
            fill_value,
            dtype=selected.dtype,
            device=selected.device,
        )
        indices = torch.tensor(destinations, dtype=torch.long, device=selected.device)
        placed.index_copy_(0, indices, selected)
        return placed
    values = list(selected)
    exemplar = values[0] if values else -1
    if _is_sequence(exemplar):
        inactive = tuple(-1 for _ in exemplar) if isinstance(exemplar, tuple) else [-1 for _ in exemplar]
    else:
        inactive = -1
    placed = [inactive for _ in range(capacity)]
    for value_row, destination in zip(values, destinations):
        placed[destination] = value_row
    if isinstance(selected, tuple):
        return tuple(placed)
    return placed


def _exact_top_k(value: Any, *, row: int, max_device_top_k: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"sampling_params.top_k[{row}] must be an integer, got {value!r}")
    top_k = int(value)
    if not 1 <= top_k <= max_device_top_k:
        raise ValueError(
            f"stochastic sampling_params.top_k[{row}]={top_k} is outside the device-supported "
            f"range [1, {max_device_top_k}]; route this request to host sampling"
        )
    return top_k
