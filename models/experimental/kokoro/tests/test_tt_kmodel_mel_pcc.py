# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro E2E spectral parity via log-mel-spectrogram PCC (mel PCC).

Scores the ported TT Kokoro vocoder against the torch CPU reference ``KModel`` audio
for the *same* text/voice/seed, across the vocoder fallback / STFT-formulation matrix
(shared with ``test_tt_kmodel_cfw2vd.py`` so both metrics score the *same* audio):

| config                        | disable_complex | stft fallback | phase fallback |
|-------------------------------|-----------------|---------------|----------------|
| ``phase_fallback``            | False           | off           | on             |
| ``stft_and_phase_fallback``   | False           | on            | on             |
| ``no_fallback``               | False           | off           | off            |
| ``dc_phase_fallback``         | True            | off           | on             |
| ``dc_no_fallback``            | True            | off           | off            |

mel PCC
-------
Sample-wise *waveform* PCC is a poor parity gate for a free-running vocoder: a tiny
phase/source drift collapses it while the speech is perceptually identical. mel PCC
compares the **magnitude** spectra instead, which discards fine phase:

1. convert each waveform to an 80-band log-mel spectrogram ``[n_mels, T]``,
2. trim both to the common number of frames,
3. return the Pearson correlation coefficient between the flattened spectrograms.

Higher is better; 1.0 means identical log-mel spectra. mel PCC is more forgiving than
waveform PCC (phase-invariant) but, unlike cFW2VD / ASR-WER, it is still frame-aligned,
so it assumes the two clips have near-equal duration (true here: same content + seed).

Run::

    pytest -s models/experimental/kokoro/tests/test_tt_kmodel_mel_pcc.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.experimental.kokoro.tests.test_tt_kmodel_asr_wer import (
    _ASR_TEXT,
    _OUTPUT_SAMPLE_RATE,
    _phonemize,
    _ref_audio,
)
from models.experimental.kokoro.tests.test_tt_kmodel_cfw2vd import (
    _CONFIGS,
    _Cfg,
    _load_kmodel,
    _tt_audio,
)
from models.experimental.kokoro.tt.tt_kmodel import preprocess_tt_kmodel

# Log-mel spectrogram parameters (24 kHz Kokoro output).
_N_FFT = 1024
_HOP_LENGTH = 256
_N_MELS = 80
# Regression floor. Measured baselines (af_heart, _ASR_TEXT, seed 0): phase-fallback
# configs ~0.993, no-fallback configs ~0.972. 0.95 gates a real spectral regression
# with comfortable headroom for every config.
MEL_PCC_MIN = 0.95


def _log_mel(waveform: torch.Tensor, sr: int) -> np.ndarray:
    """80-band log-mel spectrogram ``[n_mels, T]`` for one waveform."""
    import librosa

    audio = waveform.detach().reshape(-1).float().cpu().numpy().astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=_N_FFT, hop_length=_HOP_LENGTH, n_mels=_N_MELS, power=2.0
    )
    # log scale (PCC is invariant to the additive/scale offset, so ref choice is irrelevant).
    return librosa.power_to_db(mel, ref=np.max)


def _mel_pcc(ref_wav: torch.Tensor, gen_wav: torch.Tensor, sr: int) -> float:
    """Pearson correlation between the two log-mel spectrograms (frame-trimmed)."""
    ref_mel = _log_mel(ref_wav, sr)
    gen_mel = _log_mel(gen_wav, sr)
    t = min(ref_mel.shape[1], gen_mel.shape[1])
    r = ref_mel[:, :t].reshape(-1)
    g = gen_mel[:, :t].reshape(-1)
    if r.size == 0 or np.std(r) == 0 or np.std(g) == 0:
        return float("nan")
    return float(np.corrcoef(r, g)[0, 1])


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_ttnn_kokoro_mel_pcc(device, reset_seeds, cfg: _Cfg):
    """mel PCC between TT audio (per ``cfg``) and the matched torch reference KModel audio.

    Both waveforms are synthesized from the same text/voice/seed and the same
    ``disable_complex`` STFT formulation; the metric scores how close the TT vocoder's
    log-mel spectrum is to the reference's.
    """
    try:
        import librosa  # noqa: F401
    except Exception as exc:
        pytest.skip(f"mel PCC deps unavailable (librosa): {exc}")

    phonemes, ref_s = _phonemize(_ASR_TEXT)
    ref = _load_kmodel(cfg.disable_complex)
    params = preprocess_tt_kmodel(ref, device)

    cfg_str = (
        f"disable_complex={cfg.disable_complex}, stft_fallback={cfg.use_torch_stft_fallback}, "
        f"phase_fallback={cfg.use_torch_phase_fallback}"
    )
    logger.info("=" * 70)
    logger.info(f"KOKORO mel PCC GATE [{cfg.label}] ({cfg_str})")
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

    pcc = _mel_pcc(ref_wav, gen_wav, _OUTPUT_SAMPLE_RATE)

    ref_dur = float(ref_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    gen_dur = float(gen_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  text         : {_ASR_TEXT!r}")
    logger.info(f"  config       : {cfg_str}, seed=0, deterministic")
    logger.info(f"  ref audio    : {ref_dur:.2f}s   tt audio: {gen_dur:.2f}s   phonemes={len(phonemes)}")
    logger.info(
        f"  mel PCC      : {pcc:.4f}  floor>{MEL_PCC_MIN:.2f}  "
        f"[{'PASS' if pcc > MEL_PCC_MIN else 'LOW'}]  (higher is better; 1 = identical)"
    )

    assert np.isfinite(pcc), f"mel PCC is non-finite: {pcc}"
    assert pcc > MEL_PCC_MIN, (
        f"mel PCC {pcc:.4f} <= {MEL_PCC_MIN:.2f} [{cfg.label}]: TT log-mel spectrum diverges "
        f"catastrophically from the reference (text={_ASR_TEXT!r})"
    )

    ttnn.synchronize_device(device)
    gc.collect()
