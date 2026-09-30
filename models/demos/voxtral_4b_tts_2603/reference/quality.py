# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signal-quality scores for a RENDERED waveform: intelligibility (WER) and naturalness (MOS).

PCC against the golden says two waveforms MATCH; it cannot say either is any good. These two
scores judge the audio itself, and the e2e test scores the HF golden the same way and requires the
TT output to be no worse by a stated margin -- the thresholds come from the reference, not from an
invented absolute number.

  * intelligibility: Whisper large-v3-turbo transcribes each clip; WER against the text the
    pipeline was asked to speak, both sides through Whisper's own English normalizer.
  * naturalness: UTMOS22 (strong), a no-reference MOS predictor, via SpeechMOS.

Host-side scoring of the OUTPUT. Nothing here is in, or imported by, the forward path.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types
from functools import lru_cache

import numpy as np
import torch

ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
MOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
SCORE_RATE = 16000


def resample(wave: torch.Tensor, orig_rate: int, new_rate: int = SCORE_RATE) -> np.ndarray:
    """Band-limited polyphase resample (scipy), mono float32."""
    from math import gcd

    from scipy.signal import resample_poly

    x = wave.detach().reshape(-1).to(torch.float64).numpy()
    if int(orig_rate) == int(new_rate):
        return x.astype(np.float32)
    g = gcd(int(orig_rate), int(new_rate))
    return resample_poly(x, int(new_rate) // g, int(orig_rate) // g).astype(np.float32)


@lru_cache(maxsize=1)
def _asr():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ASR_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_ID, torch_dtype=torch.float32).eval()
    return processor, model


def transcribe(waves, rate: int, batch_size: int = 8) -> list[str]:
    processor, model = _asr()
    audio = [resample(w, rate) for w in waves]
    out = []
    for i in range(0, len(audio), batch_size):
        chunk = audio[i : i + batch_size]
        feats = processor(chunk, sampling_rate=SCORE_RATE, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(feats.input_features, language="en", task="transcribe")
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return [t.strip() for t in out]


def normalize(text: str) -> str:
    processor, _ = _asr()
    return processor.tokenizer.normalize(text)


def word_error_rate(hypothesis: str, reference: str) -> float:
    import jiwer

    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref.strip():
        raise ValueError("empty reference text")
    return float(jiwer.wer(ref, hyp if hyp.strip() else "<empty>"))


def _install_torchaudio_shim() -> None:
    """UTMOS imports torchaudio only to resample; this venv has none, and the clips arrive at 16 kHz."""
    if "torchaudio" in sys.modules:
        return
    try:
        import torchaudio  # noqa: F401

        return
    except ImportError:
        pass
    ta = types.ModuleType("torchaudio")
    fn = types.ModuleType("torchaudio.functional")
    ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    fn.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional", None)

    def _resample(wave, orig_freq, new_freq):
        if int(orig_freq) != int(new_freq):
            raise ValueError("the shim only passes 16 kHz through; resample before calling UTMOS")
        return wave

    fn.resample = _resample
    ta.functional = fn
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = fn


@lru_cache(maxsize=1)
def _mos():
    _install_torchaudio_shim()
    return torch.hub.load(MOS_HUB_REPO, "utmos22_strong", trust_repo=True).eval()


def mos(waves, rate: int) -> list[float]:
    predictor = _mos()
    scores = []
    for w in waves:
        x = torch.from_numpy(resample(w, rate)).unsqueeze(0)
        with torch.no_grad():
            scores.append(float(predictor(x, SCORE_RATE).reshape(-1)[0]))
    return scores


def score(waves, rate: int, texts) -> dict:
    """Per-clip transcript, WER and MOS, plus corpus WER (errors summed over all clips)."""
    import jiwer

    hyps = transcribe(waves, rate)
    wers = [word_error_rate(h, t) for h, t in zip(hyps, texts)]
    refs_n = [normalize(t) for t in texts]
    hyps_n = [normalize(h) if normalize(h).strip() else "<empty>" for h in hyps]
    corpus = float(jiwer.wer(refs_n, hyps_n))
    return {"transcripts": hyps, "wer": wers, "corpus_wer": corpus, "mos": mos(waves, rate)}
