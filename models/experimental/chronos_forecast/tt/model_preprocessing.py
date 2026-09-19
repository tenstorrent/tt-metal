# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side Chronos-2 input packing and (stub) weight conversion.

Step 0: build history V and future W, flatten variates onto the model batch axis.
Categorical covariates are encoded next (ordinal, or per-item target encoding when
there is a single target). InstanceNorm then standardizes each row along time
(optional arcsinh). This file does not import Amazon Chronos, and it does not run
Patch / ResidualBlock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

ArrayLike = torch.Tensor | np.ndarray | Sequence


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
    loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None


def preprocess_model_parameters(state_dict, device):
    """Map a Chronos state_dict onto device tensors."""
    raise NotImplementedError("Chronos weight conversion is not implemented yet.")


def prepare_chronos2_inputs(
    target: ArrayLike | Sequence[ArrayLike],
    *,
    prediction_length: int | None = None,
    past_covariates: ArrayLike | Sequence[ArrayLike] | None = None,
    future_covariates: ArrayLike | Sequence[ArrayLike] | None = None,
    use_target_encoding: bool = True,
    apply_instance_norm: bool = False,
    use_arcsinh: bool = False,
    instance_norm_eps: float = 1e-5,
) -> Chronos2PackedInputs:
    """Pack targets and covariates into V/W model tensors.

    Numeric covariates are passed through. Bool / string / object covariates are
    encoded: target encoding when ``use_target_encoding`` and there is one target
    row, otherwise integer ordinal codes.

    Parameters
    ----------
    target
        One series as ``(T,)`` or ``(D, T)``, or a list of those (one per series).
    prediction_length
        Future horizon H. Required when ``future_covariates`` is omitted.
    past_covariates
        Optional ``(M, T)`` / ``(T,)``, a list of M rows for one series, or a list
        of per-series payloads when ``target`` is a list. Known-future covariates
        are the last ``M_f`` rows.
    future_covariates
        Optional ``(M_f, H)`` / ``(H,)`` (or per-series list). ``M_f`` is a suffix
        of ``M``.
    use_target_encoding
        If True and each series has one target, categorical columns use per-item
        smoothed target means. Multivariate targets fall back to ordinal codes.
    apply_instance_norm
        If True, standardize each packed row along time (Amazon InstanceNorm).
        Future rows reuse the context loc/scale. Off by default so packing can
        be checked against Amazon preprocess, which does not normalize.
    use_arcsinh
        If True, apply ``arcsinh`` after standardization.
    instance_norm_eps
        Replacement scale when a row has zero variance.
    """
    target_is_batch = _is_nested_series_list(target)
    targets = _as_series_list(target, name="target")
    n_series = len(targets)
    past_per_series = _parse_covariates(
        past_covariates, n_series, target_is_batch, name="past_covariates"
    )
    future_per_series = _parse_covariates(
        future_covariates, n_series, target_is_batch, name="future_covariates"
    )

    target_2ds = [_as_variate_time(t, name="target") for t in targets]
    n_targets = target_2ds[0].shape[0]
    if any(t.shape[0] != n_targets for t in target_2ds):
        raise ValueError("all series must have the same number of target rows")

    series_lengths = [int(t.shape[-1]) for t in target_2ds]
    n_cov, n_future_cov, horizon = _covariate_layout(
        past_per_series, future_per_series, prediction_length, series_lengths
    )
    n_past_only = n_cov - n_future_cov

    encoded_past_cols, encoded_future_cols = _encode_covariate_columns(
        target_2ds=target_2ds,
        past_per_series=past_per_series,
        future_per_series=future_per_series,
        n_cov=n_cov,
        n_future_cov=n_future_cov,
        horizon=horizon,
        use_target_encoding=use_target_encoding,
    )

    context_parts: list[torch.Tensor] = []
    future_parts: list[torch.Tensor] = []
    group_ids_parts: list[torch.Tensor] = []
    target_idx_ranges: list[tuple[int, int]] = []
    row_cursor = 0

    for group_id, target_2d in enumerate(target_2ds):
        past_rows = [torch.from_numpy(encoded_past_cols[j][group_id].copy()) for j in range(n_cov)]
        future_rows = [torch.from_numpy(encoded_future_cols[j][group_id].copy()) for j in range(n_cov)]
        context = target_2d.to(dtype=torch.float32)
        if past_rows:
            context = torch.cat([context] + [row.unsqueeze(0) for row in past_rows], dim=0)
        nan_head = torch.full(
            (n_targets + n_past_only, horizon),
            float("nan"),
            dtype=torch.float32,
        )
        known_future_rows = future_rows[n_past_only:]
        if not known_future_rows:
            packed_future = nan_head
        else:
            packed_future = torch.cat(
                [nan_head] + [row.unsqueeze(0) for row in known_future_rows],
                dim=0,
            )

        n_rows = context.shape[0]
        context_parts.append(context)
        future_parts.append(packed_future)
        group_ids_parts.append(torch.full((n_rows,), group_id, dtype=torch.long))
        target_idx_ranges.append((row_cursor, row_cursor + n_targets))
        row_cursor += n_rows

    packed = Chronos2PackedInputs(
        context=_left_pad_and_cat_2d(context_parts),
        future_covariates=torch.cat(future_parts, dim=0),
        group_ids=torch.cat(group_ids_parts, dim=0),
        target_idx_ranges=target_idx_ranges,
    )
    if apply_instance_norm:
        packed = normalize_chronos2_inputs(
            packed, eps=instance_norm_eps, use_arcsinh=use_arcsinh
        )
    return packed


def instance_norm(
    x: torch.Tensor,
    loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    eps: float = 1e-5,
    use_arcsinh: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Standardize along the last dim (Amazon Chronos InstanceNorm math)."""
    orig_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    if loc_scale is None:
        loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, torch.as_tensor(eps, dtype=scale.dtype, device=scale.device), scale)
    else:
        loc, scale = loc_scale
        loc = loc.to(dtype=torch.float32)
        scale = scale.to(dtype=torch.float32)

    scaled_x = (x - loc) / scale
    if use_arcsinh:
        scaled_x = torch.arcsinh(scaled_x)
    return scaled_x.to(orig_dtype), (loc, scale)


def instance_norm_inverse(
    x: torch.Tensor,
    loc_scale: tuple[torch.Tensor, torch.Tensor],
    *,
    use_arcsinh: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Undo ``instance_norm`` with the stored loc/scale."""
    x = x.to(dtype=torch.float32)
    loc, scale = loc_scale
    loc = loc.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32)
    if use_arcsinh:
        x = torch.sinh(x)
    x = x * scale + loc
    return x if output_dtype is None else x.to(output_dtype)


def normalize_chronos2_inputs(
    packed: Chronos2PackedInputs,
    *,
    eps: float = 1e-5,
    use_arcsinh: bool = False,
) -> Chronos2PackedInputs:
    """Apply InstanceNorm to packed V, then the same loc/scale to W."""
    context, loc_scale = instance_norm(packed.context, eps=eps, use_arcsinh=use_arcsinh)
    future, _ = instance_norm(
        packed.future_covariates, loc_scale, eps=eps, use_arcsinh=use_arcsinh
    )
    return Chronos2PackedInputs(
        context=context,
        future_covariates=future,
        group_ids=packed.group_ids,
        target_idx_ranges=packed.target_idx_ranges,
        loc_scale=loc_scale,
    )


def target_encode(
    id_codes: np.ndarray,
    cat_codes: np.ndarray,
    target: np.ndarray,
    n_items: int,
    n_categories: int,
    future_id_codes: np.ndarray | None = None,
    future_cat_codes: np.ndarray | None = None,
    smooth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-item smoothed target encoding (Amazon Chronos-2 ``_target_encode`` math).

    encoded = (smooth * item_mean + category_sum) / (smooth + category_count)
    """
    mask = np.isfinite(target)
    target_masked = np.where(mask, target, 0.0)

    item_sums = np.bincount(id_codes, weights=target_masked * mask, minlength=n_items)
    item_counts = np.bincount(id_codes, weights=mask.astype(float), minlength=n_items)
    item_means = np.divide(item_sums, item_counts, out=np.zeros(n_items), where=item_counts > 0)

    n_slots = n_categories + 1
    combined_codes = id_codes * n_slots + cat_codes
    sums = np.bincount(combined_codes, weights=target_masked * mask, minlength=n_items * n_slots)
    counts = np.bincount(combined_codes, weights=mask.astype(float), minlength=n_items * n_slots)

    lookup = (smooth * np.repeat(item_means, n_slots) + sums) / (smooth + counts)
    encoded_past = lookup[combined_codes].astype(np.float32)

    encoded_future = None
    if future_id_codes is not None and future_cat_codes is not None:
        encoded_future = lookup[future_id_codes * n_slots + future_cat_codes].astype(np.float32)

    return encoded_past, encoded_future


def encode_categorical_covariate(
    past: ArrayLike,
    *,
    target: ArrayLike,
    future: ArrayLike | None = None,
    id_codes: np.ndarray | None = None,
    future_id_codes: np.ndarray | None = None,
    n_series: int = 1,
    use_target_encoding: bool = True,
    n_targets: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode one categorical covariate column to float32 past / optional future."""
    past_arr = np.asarray(past)
    target_arr = np.asarray(target, dtype=np.float32)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(1, -1)
        n_targets = 1
    elif target_arr.shape[0] != n_targets:
        n_targets = int(target_arr.shape[0])

    if id_codes is None:
        id_codes = np.zeros(len(past_arr), dtype=np.intp)
    future_arr = None if future is None else np.asarray(future)
    if future_arr is not None and future_id_codes is None:
        future_id_codes = np.zeros(len(future_arr), dtype=np.intp)
    return _encode_categorical(
        past=past_arr,
        future=future_arr,
        target=target_arr,
        id_codes=id_codes,
        future_id_codes=future_id_codes,
        n_series=n_series,
        do_target_encode=bool(use_target_encoding and n_targets == 1),
    )


def _encode_categorical(
    past: np.ndarray,
    future: np.ndarray | None,
    target: np.ndarray,
    id_codes: np.ndarray,
    future_id_codes: np.ndarray | None,
    n_series: int,
    do_target_encode: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    past_s = pd.Series(past).astype("category")
    n_real = len(past_s.dtype.categories)
    nan_slot = n_real
    n_categories = n_real + 1

    past_codes = past_s.cat.codes.to_numpy(dtype=np.intp)
    past_codes = np.where(past_codes < 0, nan_slot, past_codes)

    future_codes = None
    if future is not None:
        future_s = pd.Series(future)
        codes = past_s.dtype.categories.get_indexer(future_s).astype(np.intp)
        future_codes = np.where(codes < 0, np.where(future_s.isna().to_numpy(), nan_slot, n_categories), codes)

    if do_target_encode:
        return target_encode(
            id_codes=id_codes,
            cat_codes=past_codes,
            target=target[0],
            n_items=n_series,
            n_categories=n_categories,
            future_id_codes=future_id_codes if future_codes is not None else None,
            future_cat_codes=future_codes,
        )

    enc_past = past_codes.astype(np.float32)
    enc_future = None
    if future_codes is not None:
        enc_future = np.where(future_codes == n_categories, np.nan, future_codes).astype(np.float32)
    return enc_past, enc_future


def _encode_covariate_columns(
    *,
    target_2ds: list[torch.Tensor],
    past_per_series: list[list[np.ndarray]],
    future_per_series: list[list[np.ndarray]],
    n_cov: int,
    n_future_cov: int,
    horizon: int,
    use_target_encoding: bool,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    n_series = len(target_2ds)
    n_targets = target_2ds[0].shape[0]
    series_lengths = [int(t.shape[-1]) for t in target_2ds]
    stacked_target = np.concatenate([t.numpy() for t in target_2ds], axis=1)
    id_codes = np.repeat(np.arange(n_series), series_lengths)
    future_id_codes = np.repeat(np.arange(n_series), horizon)
    nan_future = np.full(n_series * horizon, np.nan, dtype=np.float32)
    do_target_encode = bool(use_target_encoding and n_targets == 1)

    encoded_past_cols: list[list[np.ndarray]] = []
    encoded_future_cols: list[list[np.ndarray]] = []
    for col in range(n_cov):
        stacked_past = np.concatenate([rows[col] for rows in past_per_series])
        is_known_future = col >= n_cov - n_future_cov
        stacked_future = None
        if is_known_future:
            stacked_future = np.concatenate([rows[col - (n_cov - n_future_cov)] for rows in future_per_series])

        if _is_categorical(stacked_past):
            enc_past, enc_future = _encode_categorical(
                past=stacked_past,
                future=stacked_future,
                target=stacked_target,
                id_codes=id_codes,
                future_id_codes=future_id_codes if stacked_future is not None else None,
                n_series=n_series,
                do_target_encode=do_target_encode,
            )
        else:
            enc_past = np.asarray(stacked_past, dtype=np.float32)
            enc_future = (
                np.asarray(stacked_future, dtype=np.float32) if stacked_future is not None else None
            )

        if enc_future is None:
            enc_future = nan_future
        encoded_past_cols.append(_split_by_lengths(enc_past, series_lengths))
        encoded_future_cols.append(_split_by_lengths(enc_future, [horizon] * n_series))

    return encoded_past_cols, encoded_future_cols


def _split_by_lengths(values: np.ndarray, lengths: Sequence[int]) -> list[np.ndarray]:
    out = []
    start = 0
    for length in lengths:
        out.append(np.asarray(values[start : start + length], dtype=np.float32))
        start += length
    return out


def _covariate_layout(
    past_per_series: list[list[np.ndarray]],
    future_per_series: list[list[np.ndarray]],
    prediction_length: int | None,
    series_lengths: list[int],
) -> tuple[int, int, int]:
    n_covs = {len(rows) for rows in past_per_series}
    n_futs = {len(rows) for rows in future_per_series}
    if len(n_covs) != 1 or len(n_futs) != 1:
        raise ValueError("all series must have the same number of past and future covariate rows")
    n_cov = n_covs.pop()
    n_future_cov = n_futs.pop()
    if n_future_cov > 0 and n_cov == 0:
        raise ValueError("future_covariates requires past_covariates (known-future vars must have history)")
    if n_future_cov > n_cov:
        raise ValueError(
            "future_covariates rows must be a suffix of past_covariates rows, "
            f"got M_f={n_future_cov} > M={n_cov}"
        )

    for rows, t_len in zip(past_per_series, series_lengths):
        for row in rows:
            if len(row) != t_len:
                raise ValueError(
                    f"past_covariates time length must match target, got {len(row)} vs {t_len}"
                )

    inferred: int | None = None
    for rows in future_per_series:
        for row in rows:
            h = len(row)
            if inferred is None:
                inferred = h
            elif inferred != h:
                raise ValueError(f"all future_covariates must share horizon H, got {h} vs {inferred}")

    if prediction_length is None:
        if inferred is None:
            raise ValueError("prediction_length is required when future_covariates is omitted")
        horizon = inferred
    else:
        if prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {prediction_length}")
        if inferred is not None and inferred != prediction_length:
            raise ValueError(
                "prediction_length must match future_covariates length, "
                f"got {prediction_length} vs {inferred}"
            )
        horizon = prediction_length
    return n_cov, n_future_cov, horizon


def _parse_covariates(
    value: ArrayLike | Sequence[ArrayLike] | None,
    n_series: int,
    target_is_batch: bool,
    *,
    name: str,
) -> list[list[np.ndarray]]:
    if value is None:
        return [[] for _ in range(n_series)]
    if target_is_batch:
        if not _is_nested_series_list(value) and n_series != 1:
            raise ValueError(f"{name} must be a list when target contains {n_series} series")
        items = list(value) if _is_nested_series_list(value) else [value]
        if len(items) != n_series:
            raise ValueError(f"{name} list length must match number of target series, got {len(items)} vs {n_series}")
        return [_covariate_rows(item, name=name) if item is not None else [] for item in items]
    return [_covariate_rows(value, name=name)]


def _covariate_rows(payload: ArrayLike, *, name: str) -> list[np.ndarray]:
    if _is_row_list(payload):
        rows = [np.asarray(row) for row in payload]
    else:
        arr = _as_numpy(payload)
        if arr.ndim == 1:
            rows = [arr]
        elif arr.ndim == 2:
            rows = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError(f"{name} must have shape (T,) or (n_variates, T), got {tuple(arr.shape)}")
    for row in rows:
        if row.ndim != 1:
            raise ValueError(f"{name} rows must be 1-d, got {tuple(row.shape)}")
    return rows


def _is_row_list(value: object) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray, str, bytes)) or value is None:
        return False
    if not isinstance(value, Sequence) or len(value) == 0:
        return False
    first = value[0]
    return isinstance(first, (torch.Tensor, np.ndarray, Sequence)) and not isinstance(
        first, (str, bytes, int, float)
    )


def _is_categorical(values: np.ndarray) -> bool:
    arr = np.asarray(values)
    if arr.dtype == np.bool_ or arr.dtype.kind in ("O", "U", "S"):
        return True
    return False


def _as_series_list(value: ArrayLike | Sequence[ArrayLike], *, name: str) -> list[torch.Tensor]:
    if _is_nested_series_list(value):
        series = [_to_float_tensor(item, name=name) for item in value]
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


def _to_float_tensor(value: ArrayLike, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value, dtype=np.float32))
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


def _as_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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
