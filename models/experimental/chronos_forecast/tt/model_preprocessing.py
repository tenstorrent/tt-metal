# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side Chronos-2 input packing and (stub) weight conversion.

``prepare_chronos2_inputs`` is paper Step 0: build history V and future W, then
flatten variates onto the model batch axis. It does not import Amazon Chronos,
and it does not run InstanceNorm / Patch / ResidualBlock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray | Sequence[float]


@dataclass(frozen=True)
class Chronos2PackedInputs:
    """Tensors ready for Chronos2Model.forward (host, float32).

    Shapes:
        context: (B, T)
        future_covariates: (B, H)
        group_ids: (B,)
    B is the sum of (n_targets + n_covariates) over series in the batch.
    """

    context: torch.Tensor
    future_covariates: torch.Tensor
    group_ids: torch.Tensor
    target_idx_ranges: list[tuple[int, int]]


def preprocess_model_parameters(state_dict, device):
    """Map a Chronos state_dict onto device tensors."""
    raise NotImplementedError("Chronos weight conversion is not implemented yet.")


def prepare_chronos2_inputs(
    target: ArrayLike | Sequence[ArrayLike],
    *,
    prediction_length: int | None = None,
    past_covariates: ArrayLike | Sequence[ArrayLike] | None = None,
    future_covariates: ArrayLike | Sequence[ArrayLike] | None = None,
) -> Chronos2PackedInputs:
    """Pack numeric targets and covariates into V/W model tensors.

    Parameters
    ----------
    target
        One series as ``(T,)`` or ``(D, T)``, or a list of those (one per series).
    prediction_length
        Future horizon H. Required when ``future_covariates`` is omitted.
        Must match ``future_covariates`` length when both are given.
    past_covariates
        Optional ``(M, T)`` (or ``(T,)``) aligned with ``target``, or a list.
        Known-future covariates are the last ``M_f`` rows of this matrix.
    future_covariates
        Optional ``(M_f, H)`` (or ``(H,)``). ``M_f`` must be a suffix of ``M``.
        Omitted future rows are filled with NaN (targets and past-only covs
        are always NaN in W).

    Row order per series: targets, past-only covariates, known-future covariates.
    Series with different T are left-padded with NaN along time, then concatenated.
    """
    targets = _as_series_list(target, name="target")
    n_series = len(targets)
    past_list = _align_optional_list(past_covariates, n_series, name="past_covariates")
    future_list = _align_optional_list(future_covariates, n_series, name="future_covariates")

    horizon = _resolve_horizon(prediction_length, future_list)

    context_parts: list[torch.Tensor] = []
    future_parts: list[torch.Tensor] = []
    group_ids_parts: list[torch.Tensor] = []
    target_idx_ranges: list[tuple[int, int]] = []
    row_cursor = 0

    for group_id, (tgt, past, fut) in enumerate(zip(targets, past_list, future_list)):
        target_2d = _as_variate_time(tgt, name="target")
        n_targets, time_len = target_2d.shape

        if past is None:
            past_2d = target_2d.new_empty((0, time_len))
        else:
            past_2d = _as_variate_time(past, name="past_covariates")
            if past_2d.shape[-1] != time_len:
                raise ValueError(
                    "past_covariates time length must match target, "
                    f"got {past_2d.shape[-1]} vs {time_len}"
                )

        n_cov = past_2d.shape[0]
        if fut is None:
            future_2d = target_2d.new_empty((0, horizon))
        else:
            if past is None:
                raise ValueError("future_covariates requires past_covariates (known-future vars must have history)")
            future_2d = _as_variate_time(fut, name="future_covariates")
            if future_2d.shape[-1] != horizon:
                raise ValueError(
                    "future_covariates time length must equal prediction_length, "
                    f"got {future_2d.shape[-1]} vs {horizon}"
                )

        n_future_cov = future_2d.shape[0]
        if n_future_cov > n_cov:
            raise ValueError(
                "future_covariates rows must be a suffix of past_covariates rows, "
                f"got M_f={n_future_cov} > M={n_cov}"
            )
        n_past_only = n_cov - n_future_cov

        context = torch.cat([target_2d, past_2d], dim=0)
        nan_head = torch.full(
            (n_targets + n_past_only, horizon),
            float("nan"),
            dtype=torch.float32,
        )
        if n_future_cov == 0:
            packed_future = nan_head
        else:
            packed_future = torch.cat([nan_head, future_2d.to(dtype=torch.float32)], dim=0)

        n_rows = context.shape[0]
        context_parts.append(context.to(dtype=torch.float32))
        future_parts.append(packed_future)
        group_ids_parts.append(torch.full((n_rows,), group_id, dtype=torch.long))
        target_idx_ranges.append((row_cursor, row_cursor + n_targets))
        row_cursor += n_rows

    return Chronos2PackedInputs(
        context=_left_pad_and_cat_2d(context_parts),
        future_covariates=torch.cat(future_parts, dim=0),
        group_ids=torch.cat(group_ids_parts, dim=0),
        target_idx_ranges=target_idx_ranges,
    )


def _as_series_list(value: ArrayLike | Sequence[ArrayLike], *, name: str) -> list[torch.Tensor]:
    if _is_nested_series_list(value):
        series = [ _to_float_tensor(item, name=name) for item in value ]
        if not series:
            raise ValueError(f"{name} is empty. Provide at least one series.")
        return series
    return [_to_float_tensor(value, name=name)]


def _is_nested_series_list(value: object) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return False
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    if len(value) == 0:
        return True
    first = value[0]
    return isinstance(first, (torch.Tensor, Sequence)) and not isinstance(first, (str, bytes, int, float))


def _align_optional_list(
    value: ArrayLike | Sequence[ArrayLike] | None,
    n_series: int,
    *,
    name: str,
) -> list[torch.Tensor | None]:
    if value is None:
        return [None] * n_series
    if _is_nested_series_list(value):
        items = list(value)
        if len(items) != n_series:
            raise ValueError(f"{name} list length must match number of target series, got {len(items)} vs {n_series}")
        return [None if item is None else _to_float_tensor(item, name=name) for item in items]
    if n_series != 1:
        raise ValueError(f"{name} must be a list when target contains {n_series} series")
    return [_to_float_tensor(value, name=name)]


def _resolve_horizon(prediction_length: int | None, future_list: list[torch.Tensor | None]) -> int:
    inferred: int | None = None
    for fut in future_list:
        if fut is None:
            continue
        fut_t = _as_variate_time(fut, name="future_covariates")
        h = fut_t.shape[-1]
        if inferred is None:
            inferred = h
        elif inferred != h:
            raise ValueError(f"all future_covariates must share horizon H, got {h} vs {inferred}")
    if prediction_length is None:
        if inferred is None:
            raise ValueError("prediction_length is required when future_covariates is omitted")
        return inferred
    if prediction_length <= 0:
        raise ValueError(f"prediction_length must be positive, got {prediction_length}")
    if inferred is not None and inferred != prediction_length:
        raise ValueError(
            "prediction_length must match future_covariates length, "
            f"got {prediction_length} vs {inferred}"
        )
    return prediction_length


def _to_float_tensor(value: ArrayLike, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must be at least 1-d, got scalar")
    if tensor.ndim > 2:
        raise ValueError(f"{name} must have shape (T,) or (n_variates, T), got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


def _as_variate_time(value: ArrayLike, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else _to_float_tensor(value, name=name)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"{name} must have shape (T,) or (n_variates, T), got {tuple(tensor.shape)}")


def _left_pad_and_cat_2d(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        n_variates, length = tensor.shape
        if length < max_len:
            pad = torch.full((n_variates, max_len - length), float("nan"), dtype=tensor.dtype)
            tensor = torch.cat([pad, tensor], dim=-1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)
