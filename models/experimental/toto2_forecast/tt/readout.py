# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Toto 2.0 next-patch read-out and quantile post-processing (pure torch, host side).

reference : toto2/toto2/model.py @ 9213d2c, ``Toto2Model.forecast`` (1152-1336), ``_clamp_nonfinite``
    token i predicts patch i + 1: with P context patches and h masked horizon patches (S = P + h tokens),
    tokens P - 1 .. P + h - 2 are read and token P + h - 1 is computed and discarded
    head output (9, B, V, h, 32) in asinh-normalized space; step index = token step * 9 + quantile in the 288 columns
    q = sinh(head) * scale + loc            (the horizon patches' statistics = the end-of-context ones)
    +inf / -inf -> max / min finite value along the 32 steps
    clamp to [ctx_min - K sigma_last, ctx_max + K sigma_last] (K = quantile_real_cap_k, 0 = off)
    sort over the 9 quantiles; (9, B, V, h * 32)[..., :H]
"""

from __future__ import annotations

import torch
from torch import Tensor


def readout_slots(num_context_patches: int, horizon_patches: int) -> tuple[int, int]:
    """``(start, count)`` of the tokens the head reads."""
    return num_context_patches - 1, horizon_patches


def quantile_caps(target: Tensor, mask: Tensor, scale_last: Tensor, cap_k: float) -> tuple[Tensor, Tensor] | None:
    """``(cap_min, cap_max)`` ``(B, V, 1, 1)`` from the context ``(B, V, T)``; ``mask`` is the forecast mask
    (last context patch forced observed); ``scale_last`` ``(B, V, 1)``."""
    if cap_k <= 0:
        return None
    not_obs = ~mask
    anchor_k = cap_k * scale_last
    cap_max = torch.nan_to_num(target.masked_fill(not_obs, float("-inf")).amax(-1, keepdim=True), neginf=0.0) + anchor_k
    cap_min = torch.nan_to_num(target.masked_fill(not_obs, float("inf")).amin(-1, keepdim=True), posinf=0.0) - anchor_k
    return cap_min.unsqueeze(-1), cap_max.unsqueeze(-1)


def clamp_nonfinite(vals: Tensor) -> Tensor:
    return torch.where(
        vals == float("inf"),
        torch.where(vals.isfinite(), vals, -float("inf")).amax(dim=-1, keepdim=True),
        torch.where(
            vals == -float("inf"),
            torch.where(vals.isfinite(), vals, float("inf")).amin(dim=-1, keepdim=True),
            vals,
        ),
    )


def head_columns_to_quantiles(head: Tensor, num_quantiles: int = 9) -> Tensor:
    """Head output ``(..., 32 * Q)`` (columns step-major) -> ``(Q, ..., 32)``."""
    return head.unflatten(-1, (-1, num_quantiles)).movedim(-1, 0)


def postprocess(
    head_out: Tensor, loc: Tensor, scale: Tensor, caps: tuple[Tensor, Tensor] | None, horizon: int
) -> Tensor:
    """``(9, B, V, h, 32)`` normalized -> ``(9, B, V, horizon)`` quantiles; ``loc``, ``scale`` ``(B, V, h)``."""
    q = head_out.float().sinh() * scale[..., None] + loc[..., None]
    q = clamp_nonfinite(q)
    if caps is not None:
        q = q.clamp(caps[0], caps[1])
    q = q.sort(dim=0).values
    return q.flatten(-2)[..., :horizon]
