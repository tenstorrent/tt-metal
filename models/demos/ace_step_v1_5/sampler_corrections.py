"""Host (PyTorch) sampler corrections used by the ACE-Step base sampler.

Re-implements the three corrections that live inside the official
``generate_audio`` loop (``acestep/models/base/modeling_acestep_v15_base.py``)
but are *absent* from this demo's simplified Euler loop:

1. Velocity norm clamp — prevents outlier velocity predictions.
2. Velocity EMA smoothing — low-pass filter over consecutive velocities.
3. DCW (Differential Correction in Wavelet domain) — SNR-t bias correction
   (Yu et al., CVPR 2026). Delegates to the official ``DCWCorrector`` so the
   exact paper math (Eq. 18 / 20 / 21) is preserved.

All three operate on ``[B, T, C]`` latents in float32 by convention; callers
are expected to manage dtype casts at the call site.

A matching TTNN-side implementation lives in ``ttnn_impl/sampler_corrections.py``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def apply_velocity_norm_clamp(vt: torch.Tensor, xt: torch.Tensor, threshold: float) -> torch.Tensor:
    """Clamp the velocity so ``||vt|| <= threshold * ||xt||``.

    Mirrors the official ACE-Step formula exactly::

        vt_norm = ||vt||_{dim=(1,2)}
        xt_norm = ||xt||_{dim=(1,2)} + 1e-10
        scale   = clamp(threshold * xt_norm / (vt_norm + 1e-10), max=1.0)
        vt      = vt * scale

    No-op when ``threshold <= 0``.
    """
    if threshold <= 0.0:
        return vt
    vt_norm = torch.norm(vt, dim=(1, 2), keepdim=True)
    xt_norm = torch.norm(xt, dim=(1, 2), keepdim=True) + 1e-10
    scale = torch.clamp(threshold * xt_norm / (vt_norm + 1e-10), max=1.0)
    return vt * scale


def apply_velocity_ema(vt: torch.Tensor, prev_vt: Optional[torch.Tensor], factor: float) -> torch.Tensor:
    """EMA-smooth the velocity against the previous step's velocity.

    ``vt_new = (1 - factor) * vt + factor * prev_vt`` when ``prev_vt`` is
    available; otherwise returns ``vt`` unchanged (first step). No-op when
    ``factor <= 0``.
    """
    if factor <= 0.0 or prev_vt is None:
        return vt
    return (1.0 - factor) * vt + factor * prev_vt


DcwApply = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def make_dcw_corrector(
    *,
    enabled: bool,
    mode: str,
    scaler: float,
    high_scaler: float,
    wavelet: str,
) -> Optional[DcwApply]:
    """Build a DCW corrector callable ``apply(xt, denoised, t_curr) -> xt'``.

    Delegates to the official ``DCWCorrector`` so the paper math (and the
    lazy ``pytorch_wavelets`` loader / odd-T trimming logic) is reused as-is.
    Returns ``None`` when DCW is disabled, the scaler is zero, or the
    ACE-Step repo / ``pytorch_wavelets`` are unavailable.
    """
    if not enabled:
        return None
    try:
        from acestep.models.common.dcw_correction import DCWCorrector
    except ImportError:
        return None
    corrector = DCWCorrector(
        enabled=True,
        mode=mode,
        scaler=float(scaler),
        high_scaler=float(high_scaler),
        wavelet=wavelet,
    )
    if not corrector.is_active:
        return None

    def _apply(xt: torch.Tensor, denoised: torch.Tensor, t_curr: float) -> torch.Tensor:
        return corrector.apply(xt, denoised, float(t_curr))

    return _apply
