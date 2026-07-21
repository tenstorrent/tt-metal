# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

WHISPER_MODEL_ID = "openai/whisper-large-v3"
UTMOS_SR = 16000
ECAPA2_SR = 16000
WHISPER_SR = 16000


def _as_mono_f32(wav) -> np.ndarray:
    return np.asarray(wav, dtype="float32").reshape(-1)


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Polyphase resample a 1-D signal ``sr -> target`` (no-op if already target)."""
    if sr == target:
        return wav
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), int(target))
    return resample_poly(wav, target // g, sr // g).astype("float32")


def compute_cer(wav, sr, reference_text, model_id=WHISPER_MODEL_ID, language="english"):
    """CER of ``wav`` (Whisper-large-v3 transcription) against ``reference_text``.

    Returns ``(cer, hypothesis)``. Comparison is case-insensitive and whitespace
    trimmed. ``cer`` is a fraction (0 = perfect); it can exceed 1 for very bad output.
    """
    import jiwer
    from transformers import pipeline

    if model_id not in _WHISPER:
        _WHISPER[model_id] = pipeline("automatic-speech-recognition", model=model_id)
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
    """Naturalness MOS (1-5) of ``wav`` via the UTMOS22-strong predictor."""
    global _UTMOS
    if _UTMOS is None:
        _UTMOS = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
        _UTMOS.eval()
    w = torch.from_numpy(_as_mono_f32(wav)).unsqueeze(0)  # UTMOS resamples internally
    with torch.no_grad():
        return float(_UTMOS(w, int(sr)))


def _ecapa2_embed(wav, sr) -> torch.Tensor:
    global _ECAPA2
    if _ECAPA2 is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
        _ECAPA2 = torch.jit.load(path, map_location="cpu")
        _ECAPA2.eval()
    wav16 = _resample(_as_mono_f32(wav), int(sr), ECAPA2_SR)
    t = torch.from_numpy(wav16).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        emb = _ECAPA2(t).reshape(1, -1)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def compute_secs(wav, sr, ref_wav, ref_sr):
    """Speaker Encoder Cosine Similarity between ``wav`` and ``ref_wav`` (ECAPA2).

    Returns the cosine similarity of the two L2-normalized speaker embeddings
    (higher = more similar speaker).
    """
    a = _ecapa2_embed(wav, sr)
    b = _ecapa2_embed(ref_wav, ref_sr)
    return float((a * b).sum().item())
