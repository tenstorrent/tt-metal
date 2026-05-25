# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests: TTKModel vs reference KModel — full on-device pipeline with real weights.

Why this file uses CPU fallbacks
--------------------------------
Kokoro's vocoder is dominated by two BH-BF16 precision failure points; the prosody stack
(PLBERT → DurationEncoder → predictor → TextEncoder) is precise on-device and needs **no**
fallback (stages 1–5 PCC > 0.998, verified by ``kmodel_pcc_stage_diagnostic.py``).

The two precision failure points are:

1. **SineGen phase chain** (``m_source.l_sin_gen``)
   The lerp upsample multiplies a small (<0.05) cumsum by ``2π × upsample_scale`` (= 1885
   for Kokoro). On BH, MAC ops round float32 → BF16, so 3e-5 cycle error becomes ~0.06–0.25 rad
   per frame, comparable to ``sine_amp=0.1`` — sine_wavs PCC collapses to ~0.21.
2. **STFT magnitude/phase**
   Near-zero off-frequency DFT bins (~1e-5 absolute) are rounded to 0 or sign-flipped by
   BF16 in both the conv2d and atan2 SFPU; cos(phase) PCC tops out at ~0.64 on harmonic input.

Empirical fallback sweep — single-text full pipeline (``"Hello from Tenstorrent."``)
measured by ``kmodel_fallback_comparison.py`` on Blackhole:

   Config                                              Full-pipeline PCC   Δ vs baseline
   -------------------------------------------------------------------------------------
   A. No fallback (baseline)                                  0.298452     —
   B. SineGen only                                            0.387554     +0.089
   C. STFT only                                               0.326379     +0.028
   D. SineGen + Phase                                         0.387554     +0.089   (Phase
        is REDUNDANT once SineGen-full fallback is on — sinegen replaces the entire phase
        chain in CPU, so the phase-chain-only fallback never executes.)
   E. STFT + SineGen (+ Phase, redundant) — RECOMMENDED       0.407780     +0.109
   F. STFT-conv + atan2 + Phase + SineGen (per-op STFT)       0.387412     +0.089

Interpretation
^^^^^^^^^^^^^^
* SineGen fallback contributes the largest single delta (+0.089). Without it, the
  harmonic-source path's H3 (``sine_wavs``) drops to ~0.21 PCC and poisons every downstream
  STFT bin.
* STFT fallback contributes a smaller but additive delta (+0.028 alone, +0.020 on top of
  SineGen). It restores cos(phase) PCC from ~0.64 → > 0.99 on the STFT transform output.
* The ``use_torch_phase_fallback`` flag is **subsumed by** ``use_torch_sinegen_fallback``
  (D ≡ B); we keep both flags wired through the public API so each fallback can be enabled
  in isolation when measuring decoder-only sub-paths in ``kmodel_pcc_stage_diagnostic.py``.
* ``conv+atan2`` per-op STFT (F) is slightly weaker than full ``torch.stft`` (E) because the
  full fallback also bypasses BH-BF16 padding/window accumulation, not just atan2.
* The remaining ~0.6 gap to 1.0 is the BH-BF16 ceiling on the on-device iSTFT matmul and
  decode-stack conv chain (see ``feedback_bh_bf16_stft_ceiling`` in memory and
  ``test_tt_generator_full_forward_smoke`` for the generator-only ceiling ~0.581).

Tests below assert measurement floors that bracket these empirical numbers with margin so
regressions surface immediately; per-op precision (atan2, sinegen phase, STFT magnitude)
is proved by the per-module tests under ``test_tt_torch_stft_pcc.py`` and
``test_tt_source_module_hn_nsf_pcc.py``.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))


from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tests.kmodel_pcc_stage_diagnostic import STFT_PHASE_FALLBACK_KWARGS
from models.experimental.kokoro.tt.tt_kmodel import TTKModel, preprocess_tt_kmodel

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_TEST_TEXT = "Hello from Tenstorrent."
_VOICE = "af_heart"
_LANG_CODE = "a"

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _find_checkpoint() -> Path | None:
    for p in _CKPT_CANDIDATES:
        if p.is_file():
            return p
        if p.is_dir():
            for child in p.rglob("kokoro-v1_0.pth"):
                return child
    return None


def _load_pipeline():
    try:
        from kokoro import KPipeline  # upstream package: pip install "kokoro>=0.9.2"

        return KPipeline(lang_code=_LANG_CODE, model=False)
    except ImportError:
        return None


def _phonemize(text: str) -> tuple[str, torch.Tensor]:
    """Return (phonemes, ref_s) for the first chunk of text, skipping if kokoro unavailable."""
    pipe = _load_pipeline()
    if pipe is None:
        pytest.skip("kokoro package not installed: pip install 'kokoro>=0.9.2'")

    results = list(pipe(text, voice=_VOICE))
    if not results:
        pytest.skip(f"KPipeline produced no chunks for: {text!r}")

    phonemes = results[0].phonemes
    if not phonemes:
        pytest.skip("KPipeline produced empty phonemes for first chunk.")

    pack = pipe.load_voice(_VOICE)
    ref_s = pack[len(phonemes) - 1]
    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.tensor(ref_s)
    ref_s = ref_s.float().cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    return phonemes, ref_s


@contextmanager
def _zero_noise():
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _ref_audio(ref: KModel, phonemes: str, ref_s: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
    with torch.no_grad(), _zero_noise():
        out = ref.forward(phonemes=phonemes, ref_s=ref_s, speed=speed, return_output=False)
    return out.detach().float().squeeze()


def _tt_audio(
    device,
    ref: KModel,
    params,
    phonemes: str,
    ref_s: torch.Tensor,
    *,
    use_torch_stft_fallback: bool,
    use_torch_phase_fallback: bool,
    use_torch_linear_fallback: bool = False,
    use_torch_tanh_fallback: bool = False,
    use_torch_stft_conv_fallback: bool = False,
    use_torch_atan2_fallback: bool = False,
    use_torch_sinegen_fallback: bool = False,
    use_torch_f0n_conv_fallback: bool = True,
    use_torch_f0_upsamp_fallback: bool | None = None,
    use_fp32_prosody_boundary: bool = True,
) -> torch.Tensor:
    # Construct inside _zero_noise so init-time randn_like calls match _ref_audio zeros.
    with _zero_noise():
        tt_model = TTKModel(
            device,
            ref,
            params,
            use_torch_stft_fallback=use_torch_stft_fallback,
            use_torch_stft_conv_fallback=use_torch_stft_conv_fallback,
            use_torch_atan2_fallback=use_torch_atan2_fallback,
            use_torch_phase_fallback=use_torch_phase_fallback,
            use_torch_sinegen_fallback=use_torch_sinegen_fallback,
            use_torch_linear_fallback=use_torch_linear_fallback,
            use_torch_tanh_fallback=use_torch_tanh_fallback,
            use_torch_f0n_conv_fallback=use_torch_f0n_conv_fallback,
            use_torch_f0_upsamp_fallback=use_torch_f0_upsamp_fallback,
            use_fp32_prosody_boundary=use_fp32_prosody_boundary,
        )
    out = tt_model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
    return out.audio.detach().float().squeeze()


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup(ckpt_path: Path, device) -> tuple[KModel, TTKModel, str, torch.Tensor]:
    phonemes, ref_s = _phonemize(_TEST_TEXT)

    ref = KModel(
        repo_id="hexgrad/Kokoro-82M",
        model=str(ckpt_path),
        disable_complex=True,
    ).eval()

    params = preprocess_tt_kmodel(ref, device)
    return ref, params, phonemes, ref_s


# ---------------------------------------------------------------------------
# No torch fallback path
# ---------------------------------------------------------------------------


def test_tt_kmodel_generator_no_torch_fallback_pcc(device):
    """Config A — baseline with **no** vocoder fallback. Empirical PCC ≈ 0.30.

    Uses bf16 prosody (``use_fp32_prosody_boundary=False``) and on-device F0/N conv
    (``use_torch_f0n_conv_fallback=False``) to match the historical no-fallback floor.
    P4 fp32 ``en``/F0Ntrain is enabled only for config E (see stft+phase test).

    Documents the BH-BF16 no-fallback floor. Stages 1–5 (prosody) are PCC > 0.998 here
    (see ``kmodel_pcc_stage_diagnostic.py``); the entire 0.7 deficit lives in the vocoder,
    primarily H3 SineGen sine_wavs (~0.21 PCC) and H7b STFT cos(phase) (~0.12 PCC).
    Floor (> 0.25) is set just below the measured value to catch real regressions while
    tolerating run-to-run jitter on phonemized text input.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    ref, params, phonemes, ref_s = _setup(ckpt_path, device)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    y_hat = _tt_audio(
        device,
        ref,
        params,
        phonemes,
        ref_s,
        use_torch_stft_conv_fallback=False,
        use_torch_stft_fallback=False,
        use_torch_phase_fallback=False,
        use_torch_linear_fallback=False,
        use_fp32_prosody_boundary=False,
        use_torch_f0n_conv_fallback=False,
    )

    assert y_hat.shape == y_ref.shape, (y_hat.shape, y_ref.shape)
    assert torch.isfinite(y_hat).all(), "TTKModel (no torch fallback) produced NaN/Inf"
    assert y_hat.abs().max().item() > 1e-3, "TTKModel (no torch fallback) produced ~zero output"

    _, pcc = comp_pcc(y_ref.unsqueeze(0), y_hat.unsqueeze(0), pcc=0.0)
    print(f"\nTTKModel no-torch-fallback PCC: {pcc:.6f}  phonemes={len(phonemes)}")
    assert pcc > 0.25, f"PCC {pcc:.6f} is below the no-fallback minimum floor"

    debug_mode = os.getenv("KOKORO_PCC_DEBUG_MODE", "").strip().lower()
    if debug_mode in {"stft", "phase", "both", "full"}:
        use_stft = debug_mode in {"stft", "both", "full"}
        use_phase = debug_mode in {"phase", "both", "full"}
        use_linear = debug_mode == "full"
        use_tanh = debug_mode == "full"
        y_debug = _tt_audio(
            device,
            ref,
            params,
            phonemes,
            ref_s,
            use_torch_stft_fallback=use_stft,
            use_torch_phase_fallback=use_phase,
            use_torch_linear_fallback=use_linear,
            use_torch_tanh_fallback=use_tanh,
        )
        _, pcc_debug = comp_pcc(y_ref.unsqueeze(0), y_debug.unsqueeze(0), pcc=0.0)
        print("TTKModel PCC debug: " f"mode={debug_mode}, no_fallback={pcc:.6f}, debug_mode_pcc={pcc_debug:.6f}")


# ---------------------------------------------------------------------------
# Combined stft + phase fallback path
# ---------------------------------------------------------------------------


def test_tt_kmodel_stft_and_phase_fallback_pcc(device):
    """Config E — recommended config: STFT + SineGen + Phase. Empirical PCC ≈ 0.408.

    These CPU fallbacks address the two dominant BH-BF16 precision failure points in the
    vocoder. Each is justified by a per-op test under ``test_tt_torch_stft_pcc.py`` /
    ``test_tt_source_module_hn_nsf_pcc.py`` and quantified end-to-end by
    ``kmodel_fallback_comparison.py`` (see module docstring).

    1. **STFT transform** (``use_torch_stft_fallback=True``)
       BH BF16 atan2 SFPU gives ~0.64 cos(phase) PCC on Kokoro harmonic input (near-zero
       bin sign flips). CPU ``torch.stft`` restores phase PCC > 0.99.
       End-to-end contribution: **C alone +0.028 / on top of SineGen +0.020.**

    2. **SineGen** (``use_torch_sinegen_fallback=True``)
       BH BF16 MACs amplify the small cumsum (<0.05 cycles) × 2π × upsample_scale (=1885 for
       Kokoro) → ~0.06–0.25 rad phase error per frame, comparable to ``sine_amp=0.1``.
       CPU float32 SineGen restores sine_wavs PCC > 0.99.
       End-to-end contribution: **B alone +0.089 (largest single delta).**

    3. **Phase** (``use_torch_phase_fallback=True``) — kept for explicitness but **redundant**
       when ``use_torch_sinegen_fallback=True``. The full-SineGen fallback overrides the
       phase-only fallback (D ≡ B = 0.388 in the sweep). We pass both flags because
       ``kmodel_pcc_stage_diagnostic.py`` toggles them independently to isolate sub-paths.

    Also enables ``use_torch_linear_fallback`` and ``use_torch_tanh_fallback`` so the
    m_source merge matches ref after CPU f0 upsample + SineGen (see ``DECODE_STACK_NOTES.md``).

    Floor (> 0.55): after fp32 ``en`` boundary (P4) + fp32 BiLSTM output in ``F0Ntrain`` (P5),
    captured text reaches ~0.79 PCC (was ~0.23 pre-P4).
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    ref, params, phonemes, ref_s = _setup(ckpt_path, device)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    y_hat = _tt_audio(device, ref, params, phonemes, ref_s, **STFT_PHASE_FALLBACK_KWARGS)

    assert y_hat.shape == y_ref.shape, (y_hat.shape, y_ref.shape)
    assert torch.isfinite(y_hat).all(), "TTKModel (stft+phase fallback) produced NaN/Inf"
    assert y_hat.abs().max().item() > 1e-3, "TTKModel (stft+phase fallback) produced ~zero output"

    _, pcc = comp_pcc(y_ref.unsqueeze(0), y_hat.unsqueeze(0), pcc=0.0)
    print(
        f"\nTTKModel (stft+phase fallback) PCC: {pcc:.6f}  phonemes={len(phonemes)}"
        "\n  [stft_fallback stabilizes STFT phase; "
        "sinegen_fallback stabilizes long-sequence SineGen outputs; "
        "remaining gap is outside these fallbacked ops]"
    )
    assert pcc > 0.84, (
        f"PCC {pcc:.6f} below floor (0.84) with config E fallbacks; "
        "run kmodel_pcc_stage_diagnostic.py --stft-phase-fallback --write-report"
    )


def test_tt_kmodel_conv_and_atan2_fallback_pcc(device):
    """Config F — per-op STFT alternative: conv + atan2 + Phase + SineGen. Empirical PCC ≈ 0.387.

    Per-op version of :func:`test_tt_kmodel_stft_and_phase_fallback_pcc`: instead of the full
    ``torch.stft`` bypass, only the strided conv2d (``use_torch_stft_conv_fallback``) and the
    atan2/sqrt step (``use_torch_atan2_fallback``) run on CPU — the STFT window-accumulation,
    iSTFT, and all other ops remain on-device.

    This path is weaker than full ``torch.stft`` (0.387 vs 0.408) because the on-device window
    multiplication and conv padding still propagate BF16 error; useful for measuring whether
    only the atan2 SFPU is the bottleneck. See ``test_tt_torch_stft_pcc.py`` for the per-op
    proof that conv+atan2 fallback achieves cos(phase) PCC > 0.99 on harmonic input in
    isolation, and the module docstring for the full sweep.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    ref, params, phonemes, ref_s = _setup(ckpt_path, device)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    y_hat = _tt_audio(
        device,
        ref,
        params,
        phonemes,
        ref_s,
        use_torch_stft_fallback=False,
        use_torch_stft_conv_fallback=True,
        use_torch_atan2_fallback=True,
        use_torch_phase_fallback=True,
        use_torch_sinegen_fallback=True,
    )

    assert y_hat.shape == y_ref.shape, (y_hat.shape, y_ref.shape)
    assert torch.isfinite(y_hat).all(), "TTKModel (conv+atan2+phase fallback) produced NaN/Inf"
    assert y_hat.abs().max().item() > 1e-3, "TTKModel (conv+atan2+phase fallback) produced ~zero output"

    _, pcc = comp_pcc(y_ref.unsqueeze(0), y_hat.unsqueeze(0), pcc=0.0)
    print(f"\nTTKModel (conv+atan2+phase fallback) PCC: {pcc:.6f}  phonemes={len(phonemes)}")
    assert pcc > 0.35, f"PCC {pcc:.6f} below expected floor (0.35) with conv+atan2+phase+sinegen fallback"
