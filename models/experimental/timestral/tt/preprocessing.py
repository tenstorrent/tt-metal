# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side preparation and postprocessing for one Timestral decode block (no ttnn import).

reference : models/experimental/timestral/reference/src/timestral/model.py (``_predict_context_only``)
    scaled, loc, scale = causal per-patch normalization + asinh   # float64 statistics
    tokens = [values, missing] per patch, then h forecast-query tokens
    time mask: causal over valid keys; with every key valid it is exactly ``tril``, so the
        device runs ``is_causal=True`` with no mask
    variate mask: rows of one series see each other; series are contiguous blocks of V rows
    quantiles = sinh(monotone(head(h_dec))) * anchor_scale + anchor_loc
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from models.experimental.timestral.common.timestral_src import require_timestral

require_timestral()

from timestral.config import TimestralConfig  # noqa: E402
from timestral.masks import (  # noqa: E402
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from timestral.model import monotone_quantiles  # noqa: E402
from timestral.scaler import (  # noqa: E402
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
)


@dataclass(frozen=True)
class TimestralHostInputs:
    """Host tensors for one decode block over ``R`` packed rows and ``S = L + h`` tokens."""

    tokens: Tensor  # (R, S, 2 * patch_size) float32; [values, missing] per context patch, zero decoder rows
    num_context_patches: int  # L
    horizon_patches: int  # h
    group_size: int  # V: rows per series, each series a contiguous block of rows
    cos: Tensor  # (1 or R, S, head_dim)
    sin: Tensor
    time_mask: Tensor | None  # (R, 1, S, S) bool, True = attend; None = plain causal
    variate_mask: Tensor | None  # (S * R / V, 1, V, V) bool; None = every row of a series visible
    anchor_loc: Tensor  # (R, 1) frame of the last context patch
    anchor_scale: Tensor  # (R, 1)

    @property
    def rows(self) -> int:
        return self.tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.tokens.shape[1]


def uniform_group_size(group_ids: Tensor) -> int:
    """Rows per series when every series is one contiguous block of the same size."""
    rows = group_ids.numel()
    change = torch.ones(rows, dtype=torch.bool)
    change[1:] = group_ids[1:] != group_ids[:-1]
    starts = torch.nonzero(change).flatten()
    sizes = torch.diff(torch.cat([starts, torch.tensor([rows])]))
    if torch.unique(group_ids).numel() != starts.numel() or not bool((sizes == sizes[0]).all()):
        raise NotImplementedError(
            "TT Timestral needs each series as one contiguous, equal-size block of rows "
            "(the TimestralForecaster packing); got group_ids "
            f"{group_ids.tolist() if rows <= 16 else tuple(group_ids.shape)}"
        )
    return int(sizes[0])


def rope_tables(positions: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    """``SinCosRoPE`` tables ``(rows, S, head_dim)`` for integer ``positions`` ``(rows, S)``."""
    freqs = positions[..., None].to(torch.float32) * inv_freq[None, None, :].to(torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def forecast_token_layout(
    forecast_query: Tensor, num_context_patches: int, horizon_patches: int
) -> tuple[Tensor, Tensor]:
    """``(keep, add)``, each ``(1, S, D)``: ``embed(tokens) * keep + add`` swaps the decoder rows for the query."""
    d = forecast_query.shape[-1]
    s = num_context_patches + horizon_patches
    keep = torch.zeros(1, s, d)
    keep[:, :num_context_patches] = 1.0
    add = torch.zeros(1, s, d)
    add[:, num_context_patches:] = forecast_query.to(torch.float32)
    return keep, add


def _block_diagonal(variate_mask: Tensor, group_size: int) -> Tensor:
    """``(S, 1, R, R)`` -> ``(S * R / V, 1, V, V)``: the per-series diagonal blocks."""
    s, _, r, _ = variate_mask.shape
    b = r // group_size
    m = variate_mask[:, 0].reshape(s, b, group_size, b, group_size)
    blocks = m.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)  # (S, B, V, V)
    return blocks.reshape(s * b, 1, group_size, group_size)


def prepare_inputs(
    config: TimestralConfig,
    inv_freq: Tensor,
    inputs: Tensor,
    observed: Tensor,
    group_ids: Tensor,
    horizon_patches: int,
) -> TimestralHostInputs:
    """Host half of ``TimestralModel._predict_context_only`` for packed ``(R, T)`` rows."""
    ps = config.patch_size
    rows, t = inputs.shape
    assert t % ps == 0, f"context length {t} must be a multiple of patch_size {ps}"
    assert group_ids.shape == (rows,), f"group_ids must be ({rows},), got {tuple(group_ids.shape)}"
    ll, h = t // ps, horizon_patches
    assert h >= 1, f"horizon_patches must be >= 1, got {h}"
    inputs = inputs.to(torch.float32)
    observed = observed.to(torch.bool)

    loc, scale = causal_patch_loc_scale(
        inputs.unsqueeze(1), observed.unsqueeze(1), ps, config.minimum_scale, config.stabilize_with_global
    )
    loc, scale = loc.squeeze(1), scale.squeeze(1)
    scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=config.use_asinh)
    values = scaled.reshape(rows, ll, ps)
    missing = (~observed).reshape(rows, ll, ps).to(scaled.dtype)
    tokens = torch.zeros(rows, ll + h, 2 * ps, dtype=torch.float32)
    tokens[:, :ll] = torch.cat([values, missing], dim=-1)

    obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
    key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
    all_valid = bool(key_valid.all())
    group_size = uniform_group_size(group_ids)

    time_mask = None if all_valid else build_forecast_time_mask(ll, h, key_valid, inputs.device)
    variate_mask = None
    if group_size > 1 and not all_valid:
        variate_mask = _block_diagonal(build_two_stream_variate_mask(group_ids, key_valid, h), group_size)

    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
    positions = build_forecast_positions(ll, h, first_obs)
    if bool((first_obs == first_obs[0]).all()):
        positions = positions[:1]
    cos, sin = rope_tables(positions, inv_freq)

    anchor_loc = loc.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
    anchor_scale = scale.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
    return TimestralHostInputs(
        tokens=tokens,
        num_context_patches=ll,
        horizon_patches=h,
        group_size=group_size,
        cos=cos,
        sin=sin,
        time_mask=time_mask,
        variate_mask=variate_mask,
        anchor_loc=anchor_loc,
        anchor_scale=anchor_scale,
    )


def postprocess_head(config: TimestralConfig, raw: Tensor, prepared: TimestralHostInputs) -> Tensor:
    """Head output ``(R, h, patch_size * Q)`` (normalized space) -> raw-space quantiles ``(R, h * patch_size, Q)``."""
    ps, q = config.patch_size, config.num_quantiles
    rows, h = raw.shape[:2]
    preds = raw.reshape(rows, h, ps, q).to(torch.float32)
    preds = monotone_quantiles(preds, config.quantile_knots, config.quantile_width_factor)
    preds = preds.reshape(rows, h * ps, q)
    return denormalize_quantiles(
        preds, prepared.anchor_loc.unsqueeze(-1), prepared.anchor_scale.unsqueeze(-1), config.use_asinh
    )
