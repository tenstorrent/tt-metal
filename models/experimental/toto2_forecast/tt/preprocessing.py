# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host half of ``Toto2Model.forecast`` (``decode_block_size=None``) for (B, V, T) inputs; pure torch.

reference : toto2/toto2/model.py @ 9213d2c, ``_prepare_forecast_inputs``, ``forecast`` (1092-1336), ``_sdpa_kwargs``
    full target = [context | h * 32 zeros], full mask = [mask with the last context patch forced True | False]
    loc, scale  = causal per-patch statistics (float64), then the min-obs backfill
    tokens      = [asinh((x - loc) / scale) | missing flags], (B * V, P + h, 64)
    group ids   = series id per patch; -1 on context patches with no observation
    time mask   = causal and equal group ids, per row; with one group per row it is exactly causal (None)
    variate mask = equal group ids across the V variates of an item, per token slot; None when V = 1 or all equal

Rows are ``(item, variate)`` b-major, the order of ``rearrange("(b n) s d")`` in the reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from models.experimental.toto2_forecast.tt.positions import RopeTables, rope_xpos_tables
from models.experimental.toto2_forecast.tt.readout import quantile_caps, readout_slots
from models.experimental.toto2_forecast.tt.scaler import asinh_patches, backfill_short_patches, causal_patch_loc_scale


TILE = 32


def pad_to_tile(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass(frozen=True)
class ForecastOptions:
    """The GluonTS / GIFT-Eval settings."""

    has_missing_values: bool = True
    scaler_fallback_min_obs: int = 8
    quantile_real_cap_k: float = 1e4


@dataclass(frozen=True)
class Toto2HostInputs:
    tokens: Tensor  # (R, S, 2 * patch) float32, R = B * V
    batch: int  # B
    variates: int  # V
    num_context_patches: int  # P
    horizon_patches: int  # h
    horizon: int  # H
    group_ids: Tensor  # (B, V, S) long
    time_mask: Tensor | None  # (R, S, S) bool, True = attend; None = causal
    variate_mask: Tensor | None  # (B, S, V, V) bool; None = every variate of an item visible
    rope: RopeTables  # (S, head_dim)
    loc: Tensor  # (B, V, h) float32, the horizon patches' statistics
    scale: Tensor
    caps: tuple[Tensor, Tensor] | None  # (B, V, 1, 1) each

    @property
    def rows(self) -> int:
        return self.batch * self.variates

    @property
    def seq_len(self) -> int:
        return self.num_context_patches + self.horizon_patches

    @property
    def readout(self) -> tuple[int, int]:
        return readout_slots(self.num_context_patches, self.horizon_patches)


def time_mask_from_groups(group_ids: Tensor) -> Tensor:
    """``(..., S)`` -> ``(..., S, S)`` bool: causal and equal group ids."""
    s = group_ids.shape[-1]
    causal = torch.tril(torch.ones(s, s, dtype=torch.bool))
    return causal & (group_ids[..., :, None] == group_ids[..., None, :])


def prepare_inputs(
    target: Tensor,
    target_mask: Tensor,
    series_ids: Tensor,
    horizon: int,
    *,
    patch_size: int = 32,
    head_dim: int = 64,
    options: ForecastOptions = ForecastOptions(),
) -> Toto2HostInputs:
    """``target`` (B, V, T) finite, ``target_mask`` (B, V, T) bool, ``series_ids`` (B, V) long."""
    b, v, t = target.shape
    assert t % patch_size == 0, f"context {t} must be a multiple of the patch size {patch_size}"
    target = target.to(torch.float32)
    target_mask = target_mask.to(torch.bool)
    h = math.ceil(horizon / patch_size)
    p = t // patch_size
    s = p + h
    full_target = torch.cat([target, torch.zeros(b, v, h * patch_size)], dim=-1)
    full_mask = torch.cat(
        [
            target_mask[..., :-patch_size],
            torch.ones(b, v, patch_size, dtype=torch.bool),
            torch.zeros(b, v, h * patch_size, dtype=torch.bool),
        ],
        dim=-1,
    )
    loc, scale = causal_patch_loc_scale(full_target, full_mask, patch_size)
    loc, scale = backfill_short_patches(full_target, loc, scale, full_mask, patch_size, options.scaler_fallback_min_obs)
    values = asinh_patches(full_target, full_mask, loc, scale, patch_size)
    missing = (~full_mask).to(torch.float32).unflatten(-1, (-1, patch_size))
    tokens = torch.cat([values, missing], dim=-1).reshape(b * v, s, 2 * patch_size)

    group_ids = series_ids[..., None].expand(b, v, s).clone()
    observed = full_mask[..., :t].unflatten(-1, (p, patch_size)).sum(-1)
    group_ids[..., :p][observed == 0] = -1

    time_mask = None
    if options.has_missing_values:
        flat = group_ids.reshape(b * v, s)
        if not bool((flat == flat[:, :1]).all()):
            time_mask = time_mask_from_groups(flat)
    variate_mask = None
    if v > 1:
        slots = group_ids.transpose(1, 2)  # (B, S, V)
        m = slots[..., :, None] == slots[..., None, :]
        if not bool(m.all()):
            variate_mask = m

    caps = quantile_caps(target, full_mask[..., :t], scale[..., p - 1 : p], options.quantile_real_cap_k)
    return Toto2HostInputs(
        tokens=tokens,
        batch=b,
        variates=v,
        num_context_patches=p,
        horizon_patches=h,
        horizon=horizon,
        group_ids=group_ids,
        time_mask=time_mask,
        variate_mask=variate_mask,
        rope=rope_xpos_tables(s, head_dim=head_dim),
        loc=loc[..., p:],
        scale=scale[..., p:],
        caps=caps,
    )
