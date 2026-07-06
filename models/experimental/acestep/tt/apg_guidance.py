# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance for the ACE-Step DiT denoise loop (APG, the reference default).

Faithful port of the genuine reference `apg_guidance.py` (adaptive projected guidance) used by
ACE-Step v1.5's sampling loop. The reference runs the DiT on a batch of 2 per step (conditional
context + a learned null_condition_emb) and combines the two velocity predictions with `apg_forward`:

    diff = pred_cond - pred_uncond
    momentum_buffer.update(diff)                 # momentum=-0.75 running average
    diff = momentum_buffer.running_average
    if norm_threshold>0: diff *= min(1, norm_threshold/||diff||_2 over `dims`)
    parallel, orthogonal = project(diff, pred_cond, dims)   # normalize v1 over dims
    normalized_update = orthogonal + eta*parallel           # eta=0.0
    pred_guided = pred_cond + (guidance_scale-1)*normalized_update

Two implementations:
  - `apg_forward` (host torch) — bit-faithful to the reference (double-precision project). The PCC
    ground-truth reference.
  - `apg_forward_ttnn` (pure ttnn, TRACE-CAPTURABLE) — same math in on-device ops so the CFG denoise
    loop stays trace-capturable (no host round-trip per step). Follows the tt_dit CFG idiom
    (models/tt_dit/pipelines/cfg.py uses ttnn.lerp for vanilla CFG); we extend it to APG with ttnn
    reductions. The momentum running-average is a resident device buffer updated in place per step.

Reference: modeling_acestep_v15_base.py sampling loop + apg_guidance.py (HF snapshot, ground truth).
"""

from __future__ import annotations

import torch

import ttnn


# ============================================================================
# Host torch reference (PCC ground truth) — bit-faithful to apg_guidance.py
# ============================================================================
class MomentumBuffer:
    """Running average of the guidance diff (reference default momentum = -0.75)."""

    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0.0

    def update(self, update_value: torch.Tensor):
        self.running_average = update_value + self.momentum * self.running_average


def _project(v0: torch.Tensor, v1: torch.Tensor, dims):
    """Split v0 into components parallel / orthogonal to v1 (double precision, matches reference)."""
    dtype = v0.dtype
    v0d, v1d = v0.double(), v1.double()
    v1d = torch.nn.functional.normalize(v1d, dim=dims)
    v0_parallel = (v0d * v1d).sum(dim=dims, keepdim=True) * v1d
    v0_orthogonal = v0d - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def apg_forward(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer | None = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=(1,),
) -> torch.Tensor:
    """Adaptive Projected Guidance — the reference default CFG combine. Faithful host port."""
    dims = list(dims)
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + (guidance_scale - 1) * normalized_update


def cfg_forward(pred_cond: torch.Tensor, pred_uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Vanilla CFG (linear extrapolation). Not the default; provided for comparison/ablation."""
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


# ============================================================================
# Pure-ttnn APG (trace-capturable) — same math in on-device ops
# ============================================================================
class TTMomentumBuffer:
    """On-device running average of the guidance diff. Resident buffer updated in place per step.

    running <- diff + momentum*running. Kept on device so the CFG loop is trace-safe (no host copy).
    First update (running is None) seeds with diff (momentum*0 = 0), matching the torch reference.
    """

    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = None

    def update(self, diff):
        if self.running_average is None:
            self.running_average = diff
        else:
            self.running_average = ttnn.add(diff, ttnn.multiply(self.running_average, self.momentum))
        return self.running_average


def _l2_norm_over(x, dim):
    """||x||_2 over `dim`, keepdim -> sqrt(sum(x^2, dim)). ttnn ops only."""
    return ttnn.sqrt(ttnn.sum(ttnn.square(x), dim=dim, keepdim=True))


def apg_forward_ttnn(
    pred_cond,
    pred_uncond,
    guidance_scale: float,
    momentum_buffer: TTMomentumBuffer | None = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dim: int = -2,
    eps: float = 1e-12,
):
    """Adaptive Projected Guidance in pure ttnn (trace-capturable). `dim` is the projection axis.

    For ACE-Step latents [1,1,T,64] the reference projects over the time axis (dims=[1] on its
    [B,T,C] layout) -> our axis -2. Math mirrors apg_forward exactly; precision is device bf16/fp32
    rather than the reference's double (a small PCC cost, measured honestly, not hidden).
    """
    diff = ttnn.subtract(pred_cond, pred_uncond)
    if momentum_buffer is not None:
        diff = momentum_buffer.update(diff)

    if norm_threshold > 0:
        diff_norm = _l2_norm_over(diff, dim)  # [.,.,1,C]
        # scale = min(1, norm_threshold/diff_norm); diff *= scale (broadcast over dim)
        ratio = ttnn.multiply(ttnn.reciprocal(ttnn.add(diff_norm, eps)), norm_threshold)
        scale = ttnn.minimum(ttnn.ones_like(ratio), ratio)
        diff = ttnn.multiply(diff, scale)

    # project(diff, pred_cond, dim): v1 = pred_cond / ||pred_cond||; parallel = <diff,v1> v1
    pc_norm = _l2_norm_over(pred_cond, dim)
    v1 = ttnn.multiply(pred_cond, ttnn.reciprocal(ttnn.add(pc_norm, eps)))
    dot = ttnn.sum(ttnn.multiply(diff, v1), dim=dim, keepdim=True)  # [.,.,1,C]
    parallel = ttnn.multiply(v1, dot)
    orthogonal = ttnn.subtract(diff, parallel)
    update = orthogonal if eta == 0.0 else ttnn.add(orthogonal, ttnn.multiply(parallel, eta))
    # pred_guided = pred_cond + (guidance_scale-1)*update
    return ttnn.add(pred_cond, ttnn.multiply(update, guidance_scale - 1.0))
