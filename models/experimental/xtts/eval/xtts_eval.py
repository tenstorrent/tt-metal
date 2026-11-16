# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Objective TTS evaluation metrics for XTTS-v2 synthesized audio.

Three standard TTS eval metrics, each backed by an open-source model that is
lazily loaded and module-cached (downloaded on first use from HF / torch.hub):

  * **CER**  — Character Error Rate. Transcribe the synthesized audio with
    Whisper-large-v3 and compare (``jiwer.cer``) to the input text. Lower is
    better; a proxy for pronunciation / intelligibility.
  * **UTMOS**— naturalness MOS from the UTMOS22 predictor
    (``torch.hub`` ``tarepan/SpeechMOS``). Higher (1-5) is better; an objective
    proxy for perceived speech quality.
  * **SECS** — Speaker Encoder Cosine Similarity. Cosine similarity between the
    ECAPA2 (``Jenthe/ECAPA2``) speaker embeddings of the synthesized audio and
    the target/reference speaker. Higher (-1..1) is better; speaker similarity.

The backends are heavy (Whisper-large-v3 ~3 GB, UTMOS ~400 MB, ECAPA2 ~70 MB) and
need network on first use. Each ``compute_*`` raises a clear error if its backend
is unavailable so callers can skip gracefully rather than fail hard.

All functions accept audio as a 1-D float numpy array plus its sample rate and
resample to the model's expected rate internally (16 kHz for every backend).
"""

import math

import numpy as np
import torch

# Module-level model caches (loaded once, reused across calls).
_WHISPER = {}
_UTMOS = None
_ECAPA2 = None

from models.experimental.xtts.config import (  # noqa: F401 — re-exported for callers
    EVAL_ECAPA2_REPO_ID as ECAPA2_REPO_ID,
    EVAL_ECAPA2_REVISION as ECAPA2_REVISION,
    EVAL_ECAPA2_SR as ECAPA2_SR,
    EVAL_UTMOS_HUB_REPO as UTMOS_HUB_REPO,
    EVAL_UTMOS_SR as UTMOS_SR,
    EVAL_WHISPER_MODEL_ID as WHISPER_MODEL_ID,
    EVAL_WHISPER_REVISION as WHISPER_REVISION,
    EVAL_WHISPER_SR as WHISPER_SR,
)


def _as_mono_f32(wav) -> np.ndarray:
    """Coerce audio to a 1-D float32 mono numpy array."""
    return np.asarray(wav, dtype="float32").reshape(-1)


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Polyphase resample a 1-D signal from sr to target (no-op if equal)."""
    if sr == target:
        return wav
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), int(target))
    return resample_poly(wav, target // g, sr // g).astype("float32")


def compute_cer(wav, sr, reference_text, model_id=WHISPER_MODEL_ID, language="english"):
    """Compute CER of Whisper transcription against the reference text."""
    import jiwer
    from transformers import pipeline

    if model_id not in _WHISPER:
        _WHISPER[model_id] = pipeline("automatic-speech-recognition", model=model_id, revision=WHISPER_REVISION)
    asr = _WHISPER[model_id]

    wav16 = _resample(_as_mono_f32(wav), int(sr), WHISPER_SR)
    out = asr(
        {"array": wav16, "sampling_rate": WHISPER_SR},
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    hyp = out["text"].strip()
    ref = reference_text.strip()
    cer = float(jiwer.cer(ref.lower(), hyp.lower()))
    return cer, hyp


def compute_utmos(wav, sr):
    """Return UTMOS22 naturalness MOS (1-5) for the synthesized waveform."""
    global _UTMOS
    if _UTMOS is None:
        _UTMOS = torch.hub.load(UTMOS_HUB_REPO, "utmos22_strong", trust_repo=True)
        _UTMOS.eval()
    w = torch.from_numpy(_as_mono_f32(wav)).unsqueeze(0)  # UTMOS resamples internally
    with torch.no_grad():
        return float(_UTMOS(w, int(sr)))


def _ecapa2_embed(wav, sr) -> torch.Tensor:
    """Return an L2-normalized ECAPA2 speaker embedding for the waveform."""
    global _ECAPA2
    if _ECAPA2 is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=ECAPA2_REPO_ID, filename="ecapa2.pt", revision=ECAPA2_REVISION)
        _ECAPA2 = torch.jit.load(path, map_location="cpu")
        _ECAPA2.eval()
    wav16 = _resample(_as_mono_f32(wav), int(sr), ECAPA2_SR)
    t = torch.from_numpy(wav16).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        emb = _ECAPA2(t).reshape(1, -1)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def compute_secs(wav, sr, ref_wav, ref_sr):
    """Compute ECAPA2 speaker cosine similarity between wav and ref_wav."""
    a = _ecapa2_embed(wav, sr)
    b = _ecapa2_embed(ref_wav, ref_sr)
    return float((a * b).sum().item())
