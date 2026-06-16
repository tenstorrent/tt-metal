# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro E2E speaker-identity parity via speaker-embedding cosine similarity (SECS).

Scores the ported TT Kokoro vocoder against the torch CPU reference ``KModel`` audio
for the *same* text/voice/seed, across the vocoder fallback / STFT-formulation matrix
(shared with ``test_tt_kmodel_cfw2vd.py`` / ``test_tt_kmodel_mel_pcc.py`` so all three
metrics score the *same* audio):

| config                        | disable_complex | stft fallback | phase fallback |
|-------------------------------|-----------------|---------------|----------------|
| ``phase_fallback``            | False           | off           | on             |
| ``stft_and_phase_fallback``   | False           | on            | on             |
| ``no_fallback``               | False           | off           | off            |
| ``dc_phase_fallback``         | True            | off           | on             |
| ``dc_no_fallback``            | True            | off           | off            |

Speaker cosine similarity (SECS)
--------------------------------
Sample-wise waveform PCC is a poor parity gate for a free-running vocoder. SECS asks a
different question: does the synthesized voice keep the *same speaker identity* as the
reference? It runs both waveforms through a pretrained speaker-verification x-vector
encoder (``microsoft/wavlm-base-plus-sv``) and takes the cosine similarity of the two
utterance-level embeddings:

1. resample both waveforms to 16 kHz,
2. extract the WavLM x-vector speaker embedding ``[D]`` for each,
3. return ``cos(emb_ref, emb_gen)``.

Higher is better; 1.0 means identical speaker embedding. SECS is phase- and
duration-invariant (utterance-pooled), so it isolates timbre/speaker drift rather than
sample- or frame-level error.

Run::

    pytest -s models/experimental/kokoro/tests/test_tt_kmodel_speaker_cosine.py
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

SECS_MODEL = "microsoft/wavlm-base-plus-sv"
SECS_SAMPLE_RATE = 16000
# Regression floor. Measured baselines (af_heart, _ASR_TEXT, seed 0): phase-fallback
# configs ~0.993-0.996, no-fallback configs ~0.955-0.960. 0.95 gates a real speaker-
# identity regression; note the no-fallback configs sit ~0.005 above it.
SECS_MIN = 0.95


def _resample_to_16k(waveform: torch.Tensor, src_sr: int) -> np.ndarray:
    import librosa

    audio = waveform.detach().reshape(-1).float().cpu().numpy().astype(np.float32)
    if src_sr != SECS_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=SECS_SAMPLE_RATE)
    return audio


def _speaker_embedding(audio_16k: np.ndarray, model, feature_extractor) -> torch.Tensor:
    """Utterance-level WavLM x-vector speaker embedding ``[D]``."""
    inputs = feature_extractor(audio_16k, sampling_rate=SECS_SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    # embeddings: [1, D]
    return out.embeddings.squeeze(0).float()


def _speaker_cosine(ref_wav: torch.Tensor, gen_wav: torch.Tensor, src_sr: int) -> float:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    feature_extractor = AutoFeatureExtractor.from_pretrained(SECS_MODEL)
    model = WavLMForXVector.from_pretrained(SECS_MODEL).eval()

    emb_ref = _speaker_embedding(_resample_to_16k(ref_wav, src_sr), model, feature_extractor)
    emb_gen = _speaker_embedding(_resample_to_16k(gen_wav, src_sr), model, feature_extractor)
    return float(torch.nn.functional.cosine_similarity(emb_ref, emb_gen, dim=0))


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_ttnn_kokoro_speaker_cosine(device, reset_seeds, cfg: _Cfg):
    """SECS between TT audio (per ``cfg``) and the matched torch reference KModel audio.

    Both waveforms are synthesized from the same text/voice/seed and the same
    ``disable_complex`` STFT formulation; the metric scores how close the TT vocoder's
    speaker embedding is to the reference's.
    """
    try:
        import librosa  # noqa: F401
        from transformers import AutoFeatureExtractor, WavLMForXVector  # noqa: F401
    except Exception as exc:
        pytest.skip(f"SECS deps unavailable (transformers/librosa): {exc}")

    phonemes, ref_s = _phonemize(_ASR_TEXT)
    ref = _load_kmodel(cfg.disable_complex)
    params = preprocess_tt_kmodel(ref, device)

    cfg_str = (
        f"disable_complex={cfg.disable_complex}, stft_fallback={cfg.use_torch_stft_fallback}, "
        f"phase_fallback={cfg.use_torch_phase_fallback}"
    )
    logger.info("=" * 70)
    logger.info(f"KOKORO SECS GATE [{cfg.label}] ({cfg_str})")
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

    secs = _speaker_cosine(ref_wav, gen_wav, _OUTPUT_SAMPLE_RATE)

    ref_dur = float(ref_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    gen_dur = float(gen_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  text         : {_ASR_TEXT!r}")
    logger.info(f"  config       : {cfg_str}, seed=0, deterministic")
    logger.info(f"  ref audio    : {ref_dur:.2f}s   tt audio: {gen_dur:.2f}s   phonemes={len(phonemes)}")
    logger.info(
        f"  SECS         : {secs:.4f}  floor>{SECS_MIN:.2f}  "
        f"[{'PASS' if secs > SECS_MIN else 'LOW'}]  (cosine; higher is better; 1 = same speaker)"
    )

    assert np.isfinite(secs), f"SECS is non-finite: {secs}"
    assert secs > SECS_MIN, (
        f"SECS {secs:.4f} <= {SECS_MIN:.2f} [{cfg.label}]: TT speaker identity diverges "
        f"catastrophically from the reference (text={_ASR_TEXT!r})"
    )

    ttnn.synchronize_device(device)
    gc.collect()
