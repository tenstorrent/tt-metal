"""TTNN-side sampler corrections — companion to ``sampler_corrections.py``.

Same three corrections that the official ACE-Step ``generate_audio`` loop
applies, but operating on TTNN tensors so the demo can keep the velocity
buffers on device between steps:

* :func:`ttnn_apply_velocity_norm_clamp` — pure TTNN: ``square`` + ``sum``
  + ``sqrt`` + ``mul``. One host round-trip for the scalar scale only.
* :func:`ttnn_apply_velocity_ema` — pure TTNN: ``mul`` + ``mul`` + ``add``.
  No host round-trip; ``prev_vt`` stays on device across iterations.
* :func:`ttnn_apply_dcw_correction` — host fallback. TTNN has no 1-D
  wavelet (DWT/IDWT) op, so we ``to_torch`` → official ``DCWCorrector`` →
  ``as_tensor`` once per step. Documented and intentional.

All helpers expect ``[B, T, C]`` activations in TTNN bfloat16 (the
TTNN demo's default). ``B == 1`` is assumed; the demo applies the
corrections after CFG has already merged the conditional/unconditional
batches back to a single latent.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.demos.ace_step_v1_5.sampler_corrections import DcwApply


def _scalar_from_tt(t: ttnn.Tensor) -> float:
    """Materialize a 1-element TTNN tensor as a Python float (cheap)."""
    return float(ttnn.to_torch(t).reshape(-1)[0].item())


def ttnn_apply_velocity_norm_clamp(
    vt_tt: ttnn.Tensor,
    xt_tt: ttnn.Tensor,
    threshold: float,
    eps: float = 1e-10,
) -> ttnn.Tensor:
    """Norm clamp on device — matches the official formula.

    Computes ``||vt||`` and ``||xt||`` as ``sqrt(sum(t^2))`` in TTNN, then
    derives the clamp scale on host (a single bf16 → float per step) and
    applies it as an elementwise scalar multiply on device.

    Assumes ``B == 1`` so a single scalar scale per call is correct
    (matches the demo's post-CFG batch). No-op when ``threshold <= 0``.
    """
    if threshold <= 0.0:
        return vt_tt

    vt_sq = ttnn.square(vt_tt)
    xt_sq = ttnn.square(xt_tt)
    vt_norm_sq = ttnn.sum(vt_sq)
    xt_norm_sq = ttnn.sum(xt_sq)
    vt_norm = _scalar_from_tt(vt_norm_sq) ** 0.5
    xt_norm = _scalar_from_tt(xt_norm_sq) ** 0.5 + eps
    scale = min(threshold * xt_norm / (vt_norm + eps), 1.0)
    ttnn.deallocate(vt_sq)
    ttnn.deallocate(xt_sq)
    ttnn.deallocate(vt_norm_sq)
    ttnn.deallocate(xt_norm_sq)
    if scale >= 1.0 - 1e-7:
        return vt_tt
    return ttnn.mul(vt_tt, float(scale))


def ttnn_apply_velocity_ema(
    vt_tt: ttnn.Tensor,
    prev_vt_tt: Optional[ttnn.Tensor],
    factor: float,
) -> ttnn.Tensor:
    """EMA-smooth on device: ``(1 - factor) * vt + factor * prev_vt``.

    No-op when ``factor <= 0`` or ``prev_vt_tt is None`` (first step).
    """
    if factor <= 0.0 or prev_vt_tt is None:
        return vt_tt
    cur = ttnn.mul(vt_tt, float(1.0 - factor))
    old = ttnn.mul(prev_vt_tt, float(factor))
    out = ttnn.add(cur, old)
    ttnn.deallocate(cur)
    ttnn.deallocate(old)
    return out


def ttnn_apply_dcw_correction(
    xt_tt: ttnn.Tensor,
    denoised_tt: ttnn.Tensor,
    t_curr: float,
    corrector: Optional[DcwApply],
    *,
    device,
    dtype,
    memory_config,
    layout,
) -> ttnn.Tensor:
    """Apply DCW correction via host fallback.

    TTNN has no 1-D wavelet (DWT/IDWT) op, so we round-trip the latents
    through the official ``DCWCorrector`` (which uses ``pytorch_wavelets``).
    The cost is one ``to_torch`` + one ``as_tensor`` per step — small
    compared to the DiT forward pass on the same step.

    No-op when ``corrector is None``.
    """
    if corrector is None:
        return xt_tt
    xt_h = ttnn.to_torch(xt_tt).to(torch.float32)
    den_h = ttnn.to_torch(denoised_tt).to(torch.float32)
    xt_corrected = corrector(xt_h, den_h, float(t_curr))
    return ttnn.as_tensor(
        xt_corrected.detach().contiguous().cpu().numpy(),
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def lift_to_device(
    t: torch.Tensor,
    *,
    device,
    dtype,
    layout,
    memory_config,
) -> ttnn.Tensor:
    """Convenience: torch ``[B,T,C]`` -> TTNN with the demo's standard config."""
    return ttnn.as_tensor(
        t.detach().to(dtype=torch.float32).contiguous().cpu().numpy(),
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )
