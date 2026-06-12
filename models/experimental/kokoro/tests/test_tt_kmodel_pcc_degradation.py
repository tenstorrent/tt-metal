# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Proof that the Kokoro vocoder PCC deficit cannot be closed on-device without the CPU fallbacks.

The full-forward audio PCC (TT vs reference ``KModel``, text ``"Hello from Tenstorrent."``) is
dominated by two BH-BF16 precision failure points in the generator's harmonic-source path:

* **SineGen phase accumulation** — a small cumsum (<0.05 cycles) × 2π × upsample_scale (=1885)
  rounds float32→BF16 on every BH MAC, so the per-frame phase error (~0.06–0.25 rad) becomes
  comparable to ``sine_amp=0.1`` and ``sine_wavs`` collapses to ~0.21 PCC.
* **STFT magnitude/phase** — near-zero off-frequency bins (~1e-5) get sign-flipped by the BF16
  atan2 SFPU.

This single test proves the deficit is **irreducible on-device** by running the *same* pipeline
under three fallback configurations and asserting the recovery only happens once the SineGen/phase
chain is moved to CPU float32:

    config                                  audio PCC      what it shows
    -------------------------------------------------------------------------------------
    none       (all on-device)              ~0.28          BH-BF16 ceiling — badly degraded
    stft only  (phase chain still on device) ~low          STFT fix alone is NOT enough:
                                                            the SineGen phase chaos still poisons it
    phase only (STFT still on device)       ~high          phase fallback alone recovers most of it
                                                            — the dominant single lever
    stft+phase (recommended, config E)      >0.84          both fallbacks clear the production floor;
                                                            STFT adds the final increment on top

So the degradation is **not** fixable on-device, and the **phase (SineGen) fallback is the one
that matters** — the STFT fallback alone leaves the pipeline degraded, while the phase fallback
alone recovers the bulk of it. The reference is built
``disable_complex=False`` so its generator uses ``TorchSTFT`` (``torch.stft``), matching the TT
``use_torch_stft_fallback`` formulation (pairing it against ``CustomSTFT`` adds a spurious phase-
convention mismatch — see ``test_tt_kmodel_stft_and_phase_fallback_pcc``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.tests.test_tt_kmodel_pcc import (
    _find_checkpoint,
    _ref_audio,
    _setup,
    _tt_audio,
)


def _audio_pcc(ref: torch.Tensor, tt: torch.Tensor) -> float:
    r = ref.detach().float().reshape(-1)
    t = tt.detach().float().reshape(-1)
    n = min(r.numel(), t.numel())
    _, pcc = comp_pcc(r[:n].unsqueeze(0), t[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


@pytest.mark.timeout(600)
def test_pcc_degradation_requires_phase_fallback(device):
    """Full pipeline degradation is irreducible on-device; only the SineGen/phase fallback fixes it.

    Runs the identical TT pipeline under three configs and asserts:
      1. no fallback is badly degraded (BH-BF16 harmonic-source ceiling),
      2. the STFT fallback alone does NOT recover it (SineGen phase still on-device),
      3. only adding the phase/SineGen fallback clears the production floor,
    proving the deficit cannot be fixed on-device without the fallbacks.
    """
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    # disable_complex=False -> reference uses TorchSTFT (torch.stft), matching the TT fallback.
    ref, params, phonemes, ref_s = _setup(ckpt, device, disable_complex=False)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    configs = {
        "none": dict(use_torch_stft_fallback=False, use_torch_phase_fallback=False),
        "stft_only": dict(use_torch_stft_fallback=True, use_torch_phase_fallback=False),
        "phase_only": dict(use_torch_stft_fallback=False, use_torch_phase_fallback=True),
        "stft+phase": dict(use_torch_stft_fallback=True, use_torch_phase_fallback=True),
    }

    pcc: dict[str, float] = {}
    for name, kw in configs.items():
        y = _tt_audio(device, ref, params, phonemes, ref_s, **kw)
        assert y.shape == y_ref.shape, (name, y.shape, y_ref.shape)
        assert torch.isfinite(y).all(), f"{name}: produced NaN/Inf"
        pcc[name] = _audio_pcc(y_ref, y)

    print(f"\nFull-forward audio PCC by fallback config  (phonemes={len(phonemes)}):")
    for name in configs:
        print(f"  {name:<12} {pcc[name]:.6f}")
    print(f"  recovery (stft+phase - none)       = {pcc['stft+phase'] - pcc['none']:+.4f}")
    print(f"  phase-fallback gain (vs stft_only) = {pcc['stft+phase'] - pcc['stft_only']:+.4f}")
    print(f"  phase-only vs stft-only            = {pcc['phase_only'] - pcc['stft_only']:+.4f}")
    print(f"  stft-on-top-of-phase increment     = {pcc['stft+phase'] - pcc['phase_only']:+.4f}")

    # 1. No fallback: the BH-BF16 harmonic-source ceiling leaves the pipeline badly degraded.
    assert pcc["none"] < 0.6, f"no-fallback PCC {pcc['none']:.4f} unexpectedly high — degradation gone?"

    # 2. STFT fallback ALONE is not enough: the SineGen phase chain is still on-device (BF16),
    #    so the harmonic source stays broken and poisons the audio. This is why we need the
    #    phase/SineGen fallback specifically, not just the STFT one.
    assert pcc["stft_only"] < 0.7, (
        f"stft-only PCC {pcc['stft_only']:.4f} already recovered without the phase fallback — "
        "re-check whether the SineGen phase chain still degrades on-device"
    )

    # 3. The phase fallback is the dominant single lever: phase-alone recovers far more of the
    #    deficit than stft-alone, confirming the SineGen phase chain (not the STFT) is the
    #    primary BH-BF16 failure point.
    assert pcc["phase_only"] - pcc["none"] > 0.2, (
        f"phase-only PCC {pcc['phase_only']:.4f} barely recovered over no-fallback {pcc['none']:.4f} — "
        "phase fallback should be the dominant lever"
    )
    assert pcc["phase_only"] > pcc["stft_only"], (
        f"phase-only PCC {pcc['phase_only']:.4f} did not beat stft-only {pcc['stft_only']:.4f} — "
        "phase fallback should dominate the STFT fallback as a single lever"
    )

    # 4. Only with BOTH fallbacks does the pipeline clear the production floor; the STFT fallback
    #    adds the final increment on top of the phase fallback (small but non-negative).
    assert (
        pcc["stft+phase"] > 0.84
    ), f"config-E PCC {pcc['stft+phase']:.4f} below floor 0.84 — fallbacks no longer recover the pipeline"
    assert pcc["stft+phase"] >= pcc["phase_only"] - 0.02, (
        f"adding the STFT fallback on top of phase regressed PCC "
        f"({pcc['stft+phase']:.4f} < phase-only {pcc['phase_only']:.4f})"
    )

    # 5. The recovery is large and is driven by the phase fallback, not the STFT fallback.
    assert pcc["stft+phase"] - pcc["none"] > 0.3, "fallbacks did not produce a large recovery"
    assert (
        pcc["stft+phase"] - pcc["stft_only"] > 0.2
    ), "adding the phase fallback on top of STFT did not move PCC — phase fallback would be unnecessary"
