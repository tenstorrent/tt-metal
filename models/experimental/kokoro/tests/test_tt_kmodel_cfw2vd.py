# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro E2E perceptual parity via conditional Fréchet wav2vec Distance (cFW2VD).

This gate scores the ported TT Kokoro vocoder against the torch CPU reference
``KModel`` audio for the *same* text/voice/seed, across the vocoder fallback /
STFT-formulation matrix:

| config                        | disable_complex | stft fallback | phase fallback |
|-------------------------------|-----------------|---------------|----------------|
| ``phase_fallback``            | False           | off           | on             |
| ``stft_and_phase_fallback``   | False           | on            | on             |
| ``no_fallback``               | False           | off           | off            |
| ``dc_phase_fallback``         | True            | off           | on             |
| ``dc_no_fallback``            | True            | off           | off            |

``disable_complex`` is applied to *both* the reference ``KModel`` and the TT model so
each config compares the TT port against the torch reference in the *same* STFT
formulation — the metric isolates TT-port fidelity, not a formulation mismatch.

cFW2VD (conditional Fréchet wav2vec Distance)
---------------------------------------------
Sample-wise waveform PCC is a poor parity gate for a free-running vocoder: a tiny
phase/source drift collapses PCC while the speech is perceptually identical (see the
ASR-WER gate in ``test_tt_kmodel_asr_wer.py`` for the same rationale). cFW2VD instead
compares the *distribution* of self-supervised speech features.

It is the **conditional** (paired, per-utterance) form of the Fréchet wav2vec Distance:
for the reference and generated waveforms of the *same* content we

1. resample to 16 kHz and extract frame-wise wav2vec2 hidden states ``[T, D]``,
2. model each utterance's frames as a multivariate Gaussian ``N(μ, Σ)``,
3. return the Fréchet (Wasserstein-2) distance between the two Gaussians::

       cFW2VD = ||μ_ref − μ_gen||² + Tr(Σ_ref + Σ_gen − 2 (Σ_ref Σ_gen)^½)

Lower is better; 0 means identical feature distributions. Because it is conditioned on
matched linguistic content, it is meaningful for a single utterance pair (unlike the
corpus-level unconditional FAD/FW2VD).

Run::

    pytest -s models/experimental/kokoro/tests/test_tt_kmodel_cfw2vd.py
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tests.kokoro_checkpoint import find_checkpoint
from models.experimental.kokoro.tests.test_tt_kmodel_asr_wer import (
    _ASR_TEXT,
    _OUTPUT_SAMPLE_RATE,
    _phonemize,
    _ref_audio,
)
from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig, TTKModel, preprocess_tt_kmodel

W2V_MODEL = "facebook/wav2vec2-base"
W2V_SAMPLE_RATE = 16000
# Regression ceiling (cFW2VD is a distance: lower is better, so 0.95-style similarity
# floors do not apply). Measured baselines (af_heart, _ASR_TEXT, seed 0):
#   dc_phase_fallback        0.91   (best — disable_complex phase fallback)
#   phase_fallback           1.54
#   stft_and_phase_fallback  2.06   (recommended config E)
#   no_fallback              3.64
#   dc_no_fallback           4.84   (worst — fully on-device, disable_complex)
# The fallback configs that ship in production sit at ~1-2; even the fully on-device
# configs stay under 5. 8.0 gates a real regression with ~1.65x headroom over the worst
# observed config while absorbing the run-to-run variance of the per-utterance Gaussian fit.
CFW2VD_MAX = 8.0


@dataclass(frozen=True)
class _Cfg:
    label: str
    disable_complex: bool
    use_torch_stft_fallback: bool
    use_torch_phase_fallback: bool


_CONFIGS = (
    _Cfg("phase_fallback", disable_complex=False, use_torch_stft_fallback=False, use_torch_phase_fallback=True),
    _Cfg("stft_and_phase_fallback", disable_complex=False, use_torch_stft_fallback=True, use_torch_phase_fallback=True),
    _Cfg("no_fallback", disable_complex=False, use_torch_stft_fallback=False, use_torch_phase_fallback=False),
    _Cfg("dc_phase_fallback", disable_complex=True, use_torch_stft_fallback=False, use_torch_phase_fallback=True),
    _Cfg("dc_no_fallback", disable_complex=True, use_torch_stft_fallback=False, use_torch_phase_fallback=False),
)


def _resample_to_16k(waveform: torch.Tensor, src_sr: int) -> np.ndarray:
    import librosa

    audio = waveform.detach().reshape(-1).float().cpu().numpy().astype(np.float32)
    if src_sr != W2V_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=W2V_SAMPLE_RATE)
    return audio


def _wav2vec_frames(audio_16k: np.ndarray, model, feature_extractor) -> np.ndarray:
    """Frame-wise wav2vec2 hidden states ``[T, D]`` (float64) for one utterance."""
    inputs = feature_extractor(audio_16k, sampling_rate=W2V_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values)
    # last_hidden_state: [1, T, D]
    return out.last_hidden_state.squeeze(0).double().cpu().numpy()


def _gaussian(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(μ, Σ)`` over frames; Σ regularized so its matrix sqrt is well defined."""
    mu = frames.mean(axis=0)
    # rowvar=False -> variables are columns (the D feature dims), observations are frames.
    sigma = np.cov(frames, rowvar=False)
    sigma = sigma + 1e-6 * np.eye(sigma.shape[0])
    return mu, sigma


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Fréchet (Wasserstein-2) distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    # Symmetrize to stay in the PSD manifold under numerical noise.
    sigma1 = 0.5 * (sigma1 + sigma1.T)
    sigma2 = 0.5 * (sigma2 + sigma2.T)

    sqrt_sigma1 = _matrix_sqrt_psd(sigma1)
    covmean = _matrix_sqrt_psd(sqrt_sigma1 @ sigma2 @ sqrt_sigma1)
    val = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    # cFW2VD is non-negative in theory; clamp tiny negatives from fp roundoff.
    return max(val, 0.0)


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    sqrt_evals = np.sqrt(evals)
    return (evecs * sqrt_evals) @ evecs.T


@lru_cache(maxsize=1)
def _load_w2v():
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_MODEL)
    model = Wav2Vec2Model.from_pretrained(W2V_MODEL).eval()
    return feature_extractor, model


def _cfw2vd(ref_wav: torch.Tensor, gen_wav: torch.Tensor, src_sr: int) -> float:
    feature_extractor, model = _load_w2v()

    ref_frames = _wav2vec_frames(_resample_to_16k(ref_wav, src_sr), model, feature_extractor)
    gen_frames = _wav2vec_frames(_resample_to_16k(gen_wav, src_sr), model, feature_extractor)

    mu_r, sig_r = _gaussian(ref_frames)
    mu_g, sig_g = _gaussian(gen_frames)
    return _frechet_distance(mu_r, sig_r, mu_g, sig_g)


def _load_kmodel(disable_complex: bool) -> KModel:
    ckpt = find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")
    return KModel(repo_id=KokoroConfig.repo_id, model=str(ckpt), disable_complex=disable_complex).eval()


def _tt_audio(device, ref: KModel, params, phonemes: str, ref_s: torch.Tensor, cfg: _Cfg) -> torch.Tensor:
    tt_model = TTKModel(
        device,
        ref,
        params,
        use_torch_stft_fallback=cfg.use_torch_stft_fallback,
        use_torch_phase_fallback=cfg.use_torch_phase_fallback,
        disable_complex=cfg.disable_complex,
    )
    out = tt_model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
    return out.audio.detach().float().squeeze()


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_ttnn_kokoro_cfw2vd(device, reset_seeds, cfg: _Cfg):
    """cFW2VD between TT audio (per ``cfg``) and the matched torch reference KModel audio.

    Both waveforms are synthesized from the same text/voice/seed and the same
    ``disable_complex`` STFT formulation; the metric scores how close the TT vocoder's
    wav2vec2 feature distribution is to the reference's.
    """
    try:
        import librosa  # noqa: F401
        from scipy import linalg  # noqa: F401
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model  # noqa: F401
    except Exception as exc:
        pytest.skip(f"cFW2VD deps unavailable (transformers/scipy/librosa): {exc}")

    phonemes, ref_s = _phonemize(_ASR_TEXT)
    ref = _load_kmodel(cfg.disable_complex)
    params = preprocess_tt_kmodel(ref, device)

    cfg_str = (
        f"disable_complex={cfg.disable_complex}, stft_fallback={cfg.use_torch_stft_fallback}, "
        f"phase_fallback={cfg.use_torch_phase_fallback}"
    )
    logger.info("=" * 70)
    logger.info(f"KOKORO cFW2VD GATE [{cfg.label}] ({cfg_str})")
    logger.info("=" * 70)

    # Reference (torch CPU) audio for the same content + formulation.
    torch.manual_seed(0)
    ref_wav = _ref_audio(ref, phonemes, ref_s)
    assert ref_wav.numel() > 0, "Reference KModel produced no audio samples"
    assert torch.isfinite(ref_wav).all(), "Reference waveform has non-finite samples"

    # TT audio for this config.
    torch.manual_seed(0)
    gen_wav = _tt_audio(device, ref, params, phonemes, ref_s, cfg)
    ttnn.synchronize_device(device)
    assert gen_wav.numel() > 0, f"TT Kokoro [{cfg.label}] produced no audio samples"
    assert torch.isfinite(gen_wav).all(), f"TT Kokoro [{cfg.label}] waveform has non-finite samples"

    score = _cfw2vd(ref_wav, gen_wav, _OUTPUT_SAMPLE_RATE)

    ref_dur = float(ref_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    gen_dur = float(gen_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  text         : {_ASR_TEXT!r}")
    logger.info(f"  config       : {cfg_str}, seed=0, deterministic")
    logger.info(f"  ref audio    : {ref_dur:.2f}s   tt audio: {gen_dur:.2f}s   phonemes={len(phonemes)}")
    logger.info(
        f"  cFW2VD       : {score:.4f}  ceiling<{CFW2VD_MAX:.1f}  "
        f"[{'PASS' if score < CFW2VD_MAX else 'HIGH'}]  (lower is better; 0 = identical)"
    )

    assert np.isfinite(score), f"cFW2VD is non-finite: {score}"
    assert score >= 0.0, f"cFW2VD must be non-negative, got {score}"
    assert score < CFW2VD_MAX, (
        f"cFW2VD {score:.4f} >= {CFW2VD_MAX:.1f} [{cfg.label}]: TT audio diverges "
        f"catastrophically from the reference (text={_ASR_TEXT!r})"
    )

    ttnn.synchronize_device(device)
    gc.collect()
