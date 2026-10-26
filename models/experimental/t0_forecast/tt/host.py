# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host side of one t0 forecast (pure torch / numpy, no ttnn): everything around the network.

reference : t0/forecaster.py (``predict``, ``_forecast``, ``_predict_step``, the adjustments), t0/rollout.py
(``prepare_rollout_buffer``), t0/scaler.py (``CausalScaler``), t0/patcher.py, t0/mask.py, t0/quantile.py,
t0/levels.py, t0/torch/model/layers/rope.py (``TimeAwareRotaryEmbedding``)

    prepare_inputs   series -> rollout buffer -> causal per-step scaler (fp32) -> 96-wide patch tokens,
                     masks in the device's structured form, xPos-scaled RoPE tables for q and k
    postprocess      network output (asinh space, read-out patches) -> sinh * scale + loc -> trained levels
    predict          the adjustments (trim, symmetric averaging, clip, snap), quantile interpolation and tails

The network itself is a callable ``T0HostInputs -> [rows, n_readout, patch_size, Q]`` (CPU dataflow or device).
Horizons above ``MAX_HORIZON`` (rollout) and known-future covariates are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from models.experimental.t0_forecast.tt.weights import T0Config

VALID, PAD, MISSING, CENSORED, WITHHELD = 0, 1, 2, 3, 4
TARGET, HISTORICAL, FUTURE = 0, 1, 2
MAX_HORIZON = 1024


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class Series:
    """``t0.data.TimeSeries``: per-cell values, typed mask, group ids and variate types, ``[rows, T]``."""

    values: Tensor  # float32
    mask: Tensor  # int64 MaskType
    group_ids: Tensor  # int64
    variate_type: Tensor  # int64

    @property
    def seq_len(self) -> int:
        return self.values.shape[1]


def series_from_array(context, mask=None) -> tuple[Series, tuple[int, int] | None, int, int]:
    """``ForecasterBase._series_from_array`` without covariates or group ids: ``[T]``, ``[B, T]`` or ``[B, V, T]``."""
    x = torch.as_tensor(context).to(torch.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim not in (2, 3):
        raise ValueError(f"context must be [T], [B, T] or [B, V, T]; got {tuple(x.shape)}")
    batch, width = x.shape[0], x.shape[-1]
    variates = x.shape[1] if x.ndim == 3 else 1
    rows = batch * variates
    values = x.reshape(rows, width)
    is_nan = torch.isnan(values)
    if mask is None:
        cell_mask = torch.where(is_nan, MISSING, VALID).to(torch.int64)
    else:
        cell_mask = torch.as_tensor(mask).to(torch.int64).reshape(rows, width)
        if bool(((cell_mask < VALID) | (cell_mask > CENSORED)).any()):
            raise ValueError("mask values must be VALID, PAD, MISSING or CENSORED")
        if bool((is_nan & (cell_mask == VALID)).any()):
            raise ValueError("mask marks NaN cells VALID")
    values = torch.where(is_nan, torch.zeros_like(values), values)
    groups = torch.arange(batch, dtype=torch.int64).repeat_interleave(variates)
    series = Series(
        values=values,
        mask=cell_mask,
        group_ids=groups[:, None].expand(rows, width).clone(),
        variate_type=torch.full((rows, width), TARGET, dtype=torch.int64),
    )
    batch_shape = (batch, variates) if x.ndim == 3 else None
    return series, batch_shape, rows, width


# --- adjustments around the model (t0/forecaster.py), float64 on the host ---------------------------------


def _replace_values(series: Series, values: np.ndarray, mask: np.ndarray) -> Series:
    return replace(series, values=torch.from_numpy(values.astype(np.float32)), mask=torch.from_numpy(mask))


def observed_history(series: Series, context_length: int, target_rows: Sequence[int]) -> list[np.ndarray]:
    values = series.values.numpy()[:, :context_length]
    mask = series.mask.numpy()[:, :context_length]
    return [values[r][(mask[r] == VALID) & np.isfinite(values[r])] for r in target_rows]


def trim_leading(series: Series, context_length: int) -> Series:
    """``_trim_leading_missing``: every row of a series PAD before its targets' first observation."""
    values = series.values.numpy().astype(np.float64)
    mask = series.mask.numpy().copy()
    groups = series.group_ids.numpy()[:, -1]
    targets = series.variate_type.numpy()[:, -1] == TARGET
    observed = (mask == VALID) & np.isfinite(values)
    observed[:, context_length:] = False
    for group in np.unique(groups):
        rows = groups == group
        seen = observed[rows & targets].any(axis=0)
        if seen.any():
            lead = rows[:, None] & (np.arange(series.seq_len) < int(np.argmax(seen)))[None, :]
            cells = lead & (mask != PAD)
            mask[cells] = PAD
            values[cells] = 0.0
    return _replace_values(series, values, mask)


def negated(series: Series) -> Series:
    values = series.values.numpy().astype(np.float64)
    mask = series.mask.numpy().copy()
    for row in range(values.shape[0]):
        seen = np.flatnonzero((mask[row] == VALID) & np.isfinite(values[row]))
        if seen.size == 0:
            continue
        gaps = np.flatnonzero(mask[row] == MISSING)
        gaps = gaps[gaps > seen[0]]
        if gaps.size:
            values[row, gaps] = np.interp(gaps, seen, values[row, seen])
            mask[row, gaps] = VALID
    values = np.where(mask != PAD, -np.nan_to_num(values), 0.0)
    return _replace_values(series, values, mask)


def with_mirrors(levels: Sequence[float]) -> tuple[float, ...]:
    closed = list(levels)
    for level in levels:
        if not any(math.isclose(1.0 - level, other, abs_tol=1e-9) for other in closed):
            closed.append(1.0 - level)
    return tuple(sorted(closed))


def _level_index(levels: Sequence[float], level: float) -> int:
    return next(i for i, other in enumerate(levels) if math.isclose(level, other, abs_tol=1e-9))


def averaged_rows(history: list[np.ndarray]) -> np.ndarray:
    return np.array([seen.size > 0 and seen.min() != 0 for seen in history])


def symmetric_average(positive, negative, levels, requested, history) -> np.ndarray:
    mirror = [_level_index(levels, 1.0 - level) for level in levels]
    averaged = (positive - negative[..., mirror]) / 2
    combined = np.where(averaged_rows(history)[:, None, None], averaged, positive)
    return combined[..., [_level_index(levels, level) for level in requested]]


def bound_to_history(values: np.ndarray, history, clip: bool, snap: bool) -> np.ndarray:
    for row, seen in enumerate(history):
        if not seen.size:
            continue
        if clip and seen.min() >= 0:
            values[row] = np.maximum(values[row], 0.0)
        if snap and np.array_equal(seen, np.round(seen)):
            values[row] = np.round(values[row])
    return values


# --- quantile levels (t0/levels.py, t0/quantile.py) ---------------------------------------------------------


def validate_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    levels = tuple(float(q) for q in levels)
    if not levels or any(not 0.0 < q < 1.0 for q in levels) or levels != tuple(sorted(set(levels))):
        raise ValueError(f"quantile_levels must be sorted, unique and in (0, 1); got {list(levels)}")
    return levels


def select_tail_knots(trained, requested) -> tuple[float, ...] | None:
    ordered = sorted(round(float(q), 6) for q in trained)
    lo, hi = ordered[0], ordered[-1]
    if not any((q < lo or q > hi) and 0.0 < q < 1.0 for q in requested):
        return None
    return tuple(sorted({ordered[0], ordered[1], ordered[-2], ordered[-1]}))


def rollout_quantile_levels(trained, requested) -> tuple[float, ...]:
    knots = select_tail_knots(trained, requested)
    if knots is None:
        return tuple(requested)
    interior = [q for q in requested if knots[0] <= q <= knots[-1]]
    return tuple(sorted(set(interior).union(knots)))


def interpolate_quantiles(query, source, values: Tensor) -> Tensor:
    """Linear interpolation of ``values[..., source]`` at ``query``; edge columns hold flat beyond the range."""
    f32 = np.float32
    levels = [f32(q) for q in source]
    pad_left, pad_right = bool(levels[0] > 0), bool(levels[-1] < 1)
    padded = np.array(([f32(0)] if pad_left else []) + levels + ([f32(1)] if pad_right else []), dtype=f32)
    lower, upper, weight = [], [], []
    for q in (f32(level) for level in query):
        up = min(int(np.searchsorted(padded, q, side="right")), len(padded) - 1)
        lo = up - 1
        span = padded[up] - padded[lo]
        lower.append(min(max(lo - int(pad_left), 0), len(levels) - 1))
        upper.append(min(max(up - int(pad_left), 0), len(levels) - 1))
        weight.append(0.0 if span == 0 else float(f32((q - padded[lo]) / span)))
    lo_v = values[..., torch.tensor(lower)]
    hi_v = values[..., torch.tensor(upper)]
    return lo_v + torch.tensor(weight, dtype=torch.float32) * (hi_v - lo_v)


def _extend_tails(tail_levels, knot_levels, knot_values: Tensor) -> Tensor:
    knots = [float(np.float32(q)) for q in knot_levels]
    levels = [float(np.float32(q)) for q in tail_levels]
    left = [q for q in levels if q < knots[0]]
    right = [q for q in levels if q > knots[-1]]
    columns = []
    if left:
        slope = ((knot_values[..., 1] - knot_values[..., 0]) / math.log(knots[1] / knots[0])).clamp(min=0.0)
        ratio = torch.tensor([math.log(q / knots[0]) for q in left], dtype=torch.float32)
        columns.append(knot_values[..., :1] + slope[..., None] * ratio)
    if right:
        slope = ((knot_values[..., -1] - knot_values[..., -2]) / math.log((1.0 - knots[-2]) / (1.0 - knots[-1]))).clamp(
            min=0.0
        )
        ratio = torch.tensor([math.log((1.0 - knots[-1]) / (1.0 - q)) for q in right], dtype=torch.float32)
        columns.append(knot_values[..., -1:] + slope[..., None] * ratio)
    return torch.cat(columns, dim=-1)


def extrapolate_quantiles(query, original, values: Tensor, trained) -> Tensor:
    knots = select_tail_knots(trained, query)
    if knots is None:
        return values
    left = tuple(q for q in query if q < knots[0])
    right = tuple(q for q in query if q > knots[-1])
    full = [*left, *original, *right]
    knot_values = values[..., torch.tensor([list(original).index(k) for k in knots])]
    tails = _extend_tails(left + right, knots, knot_values)
    merged = torch.cat([tails[..., : len(left)], values, tails[..., len(left) :]], dim=-1)
    return merged[..., torch.tensor([full.index(q) for q in query])]


def sanitize(values: Tensor) -> Tensor:
    values = values.to(torch.float32)
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


# --- one forward's inputs ------------------------------------------------------------------------------------


def prepare_rollout_buffer(series: Series, prediction_length: int, context_length: int, patch_size: int) -> Series:
    """``t0.rollout.prepare_rollout_buffer`` for target / historical rows (no known-future covariates)."""
    rows = series.values.shape[0]
    if bool((series.variate_type[:, -1] == FUTURE).any()):
        raise NotImplementedError("known-future covariate rows are not supported")
    pad_left = (-context_length) % patch_size
    width = round_up(prediction_length, patch_size)
    row_type = series.variate_type[:, -1:]
    row_group = series.group_ids[:, -1:]
    withheld = (row_type == TARGET) | (row_type == HISTORICAL)
    full = lambda n, v: torch.full((rows, n), v, dtype=torch.int64)
    return Series(
        values=torch.cat(
            [torch.zeros(rows, pad_left), series.values[:, :context_length], torch.zeros(rows, width)], dim=1
        ),
        mask=torch.cat(
            [
                full(pad_left, PAD),
                series.mask[:, :context_length],
                torch.where(withheld, WITHHELD, PAD).expand(rows, width),
            ],
            dim=1,
        ),
        group_ids=torch.cat(
            [full(pad_left, -1), series.group_ids[:, :context_length], row_group.expand(rows, width)], 1
        ),
        variate_type=torch.cat(
            [full(pad_left, -1), series.variate_type[:, :context_length], row_type.expand(rows, width)], 1
        ),
    )


def causal_stats(x: Tensor, invalid: Tensor, eps: float, eps_mode: str) -> tuple[Tensor, Tensor]:
    """``t0.scaler.compute_causal_stats`` in the input dtype (fp32), op for op."""
    valid = (~invalid).to(x.dtype)
    count = torch.clamp(torch.cumsum(valid, dim=-1), 1.0, None)
    masked = torch.where(invalid, torch.zeros_like(x), x)
    means = torch.cumsum(masked, dim=-1) / count
    shifted = torch.cat([torch.zeros_like(means[:, :1]), means[:, :-1]], dim=-1)
    m_2 = torch.clamp(torch.cumsum((masked - shifted) * (masked - means) * valid, dim=-1), 0.0, None)
    variance = m_2 / torch.clamp(count - 1.0, 1.0, None)
    if eps_mode == "variance_offset":
        return means, torch.sqrt(variance + eps)
    return means, torch.clamp(torch.sqrt(variance), eps, None)


def global_stats(x: Tensor, invalid: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    valid = (~invalid).to(x.dtype)
    count = valid.sum(dim=-1, keepdim=True)
    masked = torch.where(invalid, torch.zeros_like(x), x)
    mean = masked.sum(dim=-1, keepdim=True) / torch.clamp(count, 1.0, None)
    squared = torch.where(invalid, torch.zeros_like(x), (x - mean) * (x - mean))
    std = torch.clamp(torch.sqrt(squared.sum(dim=-1, keepdim=True) / torch.clamp(count, 2.0, None)), eps, None)
    return mean.expand_as(x), std.expand_as(x)


def scale_input(series: Series, cfg: T0Config) -> tuple[Tensor, Tensor, Tensor]:
    """``CausalScaler.scale_input``: ``(scaled values, loc, scale)``, each ``[rows, T]`` fp32."""
    x = series.values
    invalid = (series.mask != VALID) | torch.isnan(x)
    if not cfg.scaler_zero_no_value:
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    causal_loc, causal_scale = causal_stats(x, invalid, cfg.scaler_eps, cfg.scaler_eps_mode)
    future_loc, future_scale = global_stats(x, invalid, cfg.scaler_eps)
    real = series.group_ids >= 0
    is_future = real & (series.variate_type == FUTURE)
    is_causal = real & ((series.variate_type == TARGET) | (series.variate_type == HISTORICAL))
    loc = torch.where(is_future, future_loc, torch.where(is_causal, causal_loc, torch.zeros_like(x)))
    scale = torch.where(is_future, future_scale, torch.where(is_causal, causal_scale, torch.ones_like(x)))
    scaled = (x - loc) / scale
    if cfg.scaler_use_arcsinh:
        scaled = torch.asinh(scaled)
    if cfg.scaler_zero_no_value:
        no_value = (series.mask == PAD) | (series.mask == MISSING) | (series.mask == WITHHELD)
        scaled = torch.where(no_value, torch.zeros_like(scaled), scaled)
    return scaled, loc, scale


def reduce_patch_metadata(meta: Tensor, mask: Tensor) -> Tensor:
    """Each patch's first non-PAD value, or -1 when the patch is all PAD: ``[rows, P, ps] -> [rows, P]``."""
    is_real = mask != PAD
    first = torch.argmax(is_real.to(torch.int32), dim=-1, keepdim=True)
    reduced = torch.take_along_dim(meta, first, dim=-1).squeeze(-1)
    return torch.where(is_real.any(dim=-1), reduced, torch.full((), -1, dtype=reduced.dtype))


def build_attention_masks(group_ids: Tensor, variate_type: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """``t0.mask.build_attention_masks`` on patched ``[rows, P, ps]`` fields: time ``[rows, P, P]``, group
    ``[P, rows, rows]`` (True = attend). Dense; the device uses the structured form of ``prepare_inputs``."""
    pg = reduce_patch_metadata(group_ids, mask)
    pv = reduce_patch_metadata(variate_type, mask)
    attendable = (mask != PAD).any(dim=-1)
    valid = pg >= 0
    future = pv == FUTURE
    same = (pg[:, :, None] == pg[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    pos = torch.arange(pg.shape[1])
    causal = pos[:, None] >= pos[None, :]
    fq = future[:, :, None]
    time = ((fq & same) | (~fq & same & causal)) & attendable[:, None, :]
    ids, ok, fut = pg.T, valid.T, future.T
    group = (ids[:, :, None] == ids[:, None, :]) & ok[:, :, None] & ok[:, None, :]
    group = group & ~(fut[:, :, None] & ~fut[:, None, :])
    return time, group


def rope_tables(num_patches: int, cfg: T0Config) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """xPos-scaled RoPE for ``num_patches`` positions in t0's interleaved pair layout: ``(cos_q, sin_q, cos_k,
    sin_k)``, each ``[S, head_dim]``. ``q_rot = q * cos_q + rot(q) * sin_q`` with ``rot`` the pair rotation
    ``(a, b) -> (-b, a)``; k likewise with ``zeta^-power``. The xPos centre is ``floor((S - 1) / 2)``
    (``t.max() // 2``), so the tables depend on S; the centre cancels in ``q . k`` but sets the magnitudes."""
    hd = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2)[: hd // 2].float() / hd))
    seq = torch.arange(num_patches, dtype=torch.float32)
    freqs = torch.einsum("..., f -> ... f", seq, inv_freq).repeat_interleave(2, dim=-1)
    zeta = (torch.arange(0, hd, 2) + 0.4 * hd) / (1.4 * hd)
    power = (seq - seq.max(-1).values.unsqueeze(-1) // 2) / cfg.xpos_scale_base
    scale = zeta ** power[:, None]
    if cfg.xpos_scale_layout == "halves":
        scale = torch.cat((scale, scale), dim=-1)
    elif cfg.xpos_scale_layout == "interleaved":
        scale = scale.repeat_interleave(2, dim=-1)
    else:
        raise ValueError(f"unknown xpos_scale_layout {cfg.xpos_scale_layout!r}")
    cos, sin = freqs.cos(), freqs.sin()
    inv = scale**-1
    return cos * scale, sin * scale, cos * inv, sin * inv


def pair_rotation(n: int) -> Tensor:
    """``(n, n)`` matrix with ``t @ R`` mapping each pair ``(t_2i, t_2i+1)`` to ``(-t_2i+1, t_2i)``."""
    r = torch.zeros(n, n)
    r[torch.arange(0, n, 2), torch.arange(1, n, 2)] = 1.0
    r[torch.arange(1, n, 2), torch.arange(0, n, 2)] = -1.0
    return r


def monotone_matrix(n_quantiles: int, patch_size: int) -> tuple[Tensor, Tensor]:
    """``(first, cumulative)``: ``first`` is the ``[patch_size * Q]`` indicator of each step's level 0, and
    ``kron(I, U)`` sums ``[raw_0, softplus(raw_1..)]`` into the quantiles of every step."""
    u = torch.triu(torch.ones(n_quantiles, n_quantiles))
    first = torch.zeros(patch_size, n_quantiles)
    first[:, 0] = 1.0
    return first.reshape(-1), torch.kron(torch.eye(patch_size), u)


@dataclass(frozen=True)
class T0HostInputs:
    """One forward pass, prepared on the host."""

    rows: int  # R, every row of the window (targets first)
    num_patches: int  # S, context + withheld patches
    context_patches: int
    readout_start: int  # first output patch read: context_patches - 1
    readout_count: int  # horizon / patch_size
    group_size: int  # G: rows of one series, contiguous
    tokens: Tensor  # [R, S, 3 * patch_size] fp32
    patch_type: Tensor  # [R, S] int64, the type embedding index of each patch
    time_mask: Tensor | None  # [R, S, S] bool; None = plain causal
    group_mask: Tensor | None  # [S, R / G, G, G] bool; None = all G rows of a series see each other
    real_tokens: Tensor  # [R, S] bool: patches that are not all PAD
    rope: tuple[Tensor, Tensor, Tensor, Tensor]  # cos_q, sin_q, cos_k, sin_k, [S, head_dim]
    loc: Tensor  # [R, readout_count] at each read-out patch's last step
    scale: Tensor
    horizon: int  # rounded to whole patches
    target_rows: tuple[int, ...]
    batch_shape: tuple[int, int] | None = None  # (B, V) for [B, V, T] input
    prediction_length: int = 0


def _group_size(row_groups: Tensor) -> int:
    """Rows of one series must be contiguous and equally many; returns that count."""
    change = torch.ones(len(row_groups), dtype=torch.bool)
    change[1:] = row_groups[1:] != row_groups[:-1]
    starts = torch.nonzero(change).flatten()
    sizes = torch.diff(torch.cat([starts, torch.tensor([len(row_groups)])]))
    if len(set(sizes.tolist())) != 1 or len(set(row_groups.tolist())) != len(starts):
        raise NotImplementedError("rows of a series must be contiguous and every series must have as many rows")
    return int(sizes[0])


def structured_masks(groups: Tensor, vtype: Tensor, mask: Tensor):
    """Patched ``[R, S, ps]`` fields -> ``(time_mask, group_mask, real, group_size)`` in the device's form.

    ``time_mask`` ``[R, S, S]`` or None (plain causal), ``group_mask`` ``[S, R / G, G, G]`` or None (every row
    of a series sees the others), ``real`` ``[R, S]`` marks patches that are not all PAD. All-PAD queries
    attend nothing in the reference (torch returns 0 there); here they read themselves, so every token stays
    finite. No real query ever reads them, so real tokens are unchanged."""
    rows, s, _ = mask.shape
    real = (mask != PAD).any(dim=-1)
    pg = reduce_patch_metadata(groups, mask)
    pv = reduce_patch_metadata(vtype, mask)
    row_ids = torch.where(real, pg, torch.iinfo(torch.int64).min).max(dim=1).values
    group_size = _group_size(row_ids)
    clean = bool(real.all()) and bool((pg == row_ids[:, None]).all()) and not bool((pv == FUTURE).any())
    if clean:
        return None, None, real, group_size
    time, group = build_attention_masks(groups, vtype, mask)
    time_mask = time | (~real[:, :, None] & torch.eye(s, dtype=torch.bool))
    g = group_size
    blocks = group.reshape(s, rows // g, g, rows // g, g).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)
    blocks = blocks | (~real.T.reshape(s, rows // g, g)[..., None] & torch.eye(g, dtype=torch.bool))
    group_mask = None if bool(blocks.all()) else blocks.contiguous()
    return time_mask, group_mask, real, group_size


def prepare_window(cfg: T0Config, window: Series, horizon: int, target_rows: Sequence[int]) -> T0HostInputs:
    """``ForecasterBase._predict_step`` up to the network: scale, patch, masks, RoPE tables."""
    ps = cfg.patch_size
    rows, width = window.values.shape
    scaled, loc, scale = scale_input(window, cfg)
    s = width // ps
    patch = lambda t: t.reshape(rows, s, ps)
    values, mask, groups, vtype = patch(scaled), patch(window.mask), patch(window.group_ids), patch(window.variate_type)
    t_index = (torch.arange(ps, dtype=torch.float32) / ps).expand(rows, s, ps)
    tokens = torch.cat([values, t_index, (mask == VALID).to(torch.float32)], dim=-1)
    patch_type = reduce_patch_metadata(vtype, mask).clamp(min=0)
    time_mask, group_mask, real, group_size = structured_masks(groups, vtype, mask)
    context_patches = (width - horizon) // ps
    start, count = context_patches - 1, horizon // ps
    ends = slice(ps - 1, None, ps)
    return T0HostInputs(
        rows=rows,
        num_patches=s,
        context_patches=context_patches,
        readout_start=start,
        readout_count=count,
        group_size=group_size,
        tokens=tokens,
        patch_type=patch_type,
        time_mask=time_mask,
        group_mask=group_mask,
        real_tokens=real,
        rope=rope_tables(s, cfg),
        loc=loc[:, ends][:, start : start + count].contiguous(),
        scale=scale[:, ends][:, start : start + count].contiguous(),
        horizon=horizon,
        target_rows=tuple(target_rows),
    )


def prepare_series(
    cfg: T0Config, series: Series, prediction_length: int, context_length: int, target_rows
) -> T0HostInputs:
    """``rollout`` up to its single pass (horizon <= ``MAX_HORIZON``)."""
    if prediction_length > MAX_HORIZON:
        raise NotImplementedError(f"horizon {prediction_length} > {MAX_HORIZON} needs the rollout")
    buffer = prepare_rollout_buffer(series, prediction_length, context_length, cfg.patch_size)
    horizon = round_up(prediction_length, cfg.patch_size)
    width = round_up(context_length, cfg.patch_size) + horizon
    window = replace(
        buffer,
        values=buffer.values[:, :width],
        mask=buffer.mask[:, :width],
        group_ids=buffer.group_ids[:, :width],
        variate_type=buffer.variate_type[:, :width],
    )
    return prepare_window(cfg, window, horizon, target_rows)


def rescale(cfg: T0Config, normalized: Tensor, prepared: T0HostInputs) -> Tensor:
    """Network output ``[R, n_readout, ps, Q]`` (asinh space) -> data space ``[targets, horizon, Q]`` (fp32)."""
    q = normalized.to(torch.float32)
    values = torch.sinh(q) if cfg.scaler_use_arcsinh else q
    out = values * prepared.scale[:, :, None, None] + prepared.loc[:, :, None, None]
    out = out.reshape(prepared.rows, prepared.horizon, cfg.n_quantiles)
    return out[torch.tensor(prepared.target_rows)]


Network = Callable[[T0HostInputs], Tensor]


def forecast_levels(
    cfg: T0Config, network: Network, series: Series, horizon: int, context_length: int, levels, target_rows
) -> Tensor:
    """``ForecasterBase._forecast``: one forward, ``[targets, horizon, len(levels)]``."""
    rollout_levels = rollout_quantile_levels(cfg.quantile_levels, levels)
    prepared = prepare_series(cfg, series, horizon, context_length, target_rows)
    block = rescale(cfg, network(prepared), prepared)
    out = interpolate_quantiles(rollout_levels, cfg.quantile_levels, block)[:, :horizon]
    return extrapolate_quantiles(levels, rollout_levels, sanitize(out), cfg.quantile_levels)


def predict(
    cfg: T0Config,
    network: Network,
    context,
    horizon: int,
    quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
    *,
    mask=None,
    trim_leading_missing: bool = False,
    symmetric_averaging: bool = False,
    clip_nonnegative: bool = False,
    snap_integers: bool = False,
) -> Tensor:
    """``T0Forecaster.predict`` for array inputs: ``[B, horizon, Q]`` or ``[B, V, horizon, Q]``."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    levels = validate_quantile_levels(quantile_levels)
    series, batch_shape, n_targets, context_length = series_from_array(context, mask)
    target_rows = list(range(n_targets))
    if trim_leading_missing:
        series = trim_leading(series, context_length)
    run_levels = with_mirrors(levels) if symmetric_averaging else levels
    out = forecast_levels(cfg, network, series, horizon, context_length, run_levels, target_rows)
    if symmetric_averaging or clip_nonnegative or snap_integers:
        history = observed_history(series, context_length, target_rows)
        values = out.numpy().astype(np.float64)
        if symmetric_averaging:
            neg = (
                forecast_levels(cfg, network, negated(series), horizon, context_length, run_levels, target_rows)
                .numpy()
                .astype(np.float64)
                if averaged_rows(history).any()
                else values
            )
            values = symmetric_average(values, neg, run_levels, levels, history)
        values = bound_to_history(values, history, clip_nonnegative, snap_integers)
        out = torch.from_numpy(values.astype(np.float32))
    if batch_shape is not None:
        out = out.reshape(*batch_shape, *out.shape[1:])
    return out
