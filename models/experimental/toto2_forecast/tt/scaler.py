# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Toto 2.0 causal per-patch scaler, host side (pure torch, float64 statistics).

reference : toto2/toto2/model.py @ 9213d2c, ``PatchedCausalStdScaler`` (139-206), ``backfill_short_patches`` (104-130)
    n_t  = sum_{s<=t} m_s;  mu_t = sum m x / max(1, n_t)
    M2_t = sum_{s<=t} m_s (x_s - mu_{s-1}) (x_s - mu_s)           # Welford in cumsum form, mu_{-1} = 0
    sigma_t = max(sqrt(M2_t / max(1, n_t - 1)), 1e-6)
    patch k uses (mu, sigma) at its last step, so it includes itself; masked steps leave n and M2 unchanged
    backfill (min_obs > 0): leading patches with fewer than min_obs cumulative observations take the mean and
        unbiased std of the first min_obs observed points
    z = asinh((x - mu) / sigma) at observed steps, 0 elsewhere

The statistics are returned per patch, ``(..., P)``; only those values reach the model.
"""

from __future__ import annotations

import torch
from torch import Tensor


def causal_patch_loc_scale(
    target: Tensor, mask: Tensor, patch_size: int, *, correction: int = 1, minimum_scale: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """``(loc, scale)`` per patch, ``(..., T / patch_size)``, in ``target.dtype`` (float64 inside)."""
    data = target.to(torch.float64)
    cum = (data * mask).cumsum(dim=-1)
    denominator = mask.cumsum(dim=-1).clamp_min(1)
    loc = cum / denominator
    prev = torch.cat([torch.zeros_like(loc[..., :1]), loc[..., :-1]], dim=-1)
    m2 = torch.cumsum((data - prev) * (data - loc) * mask, dim=-1)
    scale = torch.sqrt(m2 / (denominator - correction).clamp(min=1)).clamp(min=minimum_scale)
    last = slice(patch_size - 1, None, patch_size)
    return loc[..., last].to(target.dtype), scale[..., last].to(target.dtype)


def backfill_short_patches(
    target: Tensor, loc: Tensor, scale: Tensor, mask: Tensor, patch_size: int, min_obs: int
) -> tuple[Tensor, Tensor]:
    """Per-patch ``(loc, scale)`` with the leading short patches replaced by the first-``min_obs`` donor;
    computed in ``target.dtype`` as the reference does."""
    if min_obs <= 0:
        return loc, scale
    in_first_n = (mask.cumsum(dim=-1) <= min_obs) & mask
    n = in_first_n.sum(dim=-1, keepdim=True).clamp(min=1)
    donor_loc = (target * in_first_n).sum(dim=-1, keepdim=True) / n
    donor_var = (((target - donor_loc) * in_first_n) ** 2).sum(dim=-1, keepdim=True) / (n - 1).clamp(min=1)
    donor_scale = donor_var.sqrt().clamp(min=1e-6)
    below = mask.unflatten(-1, (-1, patch_size)).sum(dim=-1).cumsum(dim=-1) < min_obs
    return torch.where(below, donor_loc, loc), torch.where(below, donor_scale, scale)


def asinh_patches(target: Tensor, mask: Tensor, loc: Tensor, scale: Tensor, patch_size: int) -> Tensor:
    """``(..., T)`` -> ``(..., T / patch_size, patch_size)`` of ``asinh((x - loc) / scale)``, 0 where masked;
    ``loc`` and ``scale`` per patch."""
    x = target.unflatten(-1, (-1, patch_size))
    z = (x - loc[..., None]) / scale[..., None]
    return torch.where(mask.unflatten(-1, (-1, patch_size)), z, torch.zeros_like(z)).asinh()
