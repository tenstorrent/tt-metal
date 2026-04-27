# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Speaker similarity for two in-memory audio waveforms."""

from __future__ import annotations

import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_SPEAKER_ENCODER = "microsoft/wavlm-base-plus-sv"
SPEAKER_EMBEDDING_SAMPLE_RATE = 16000


def _compute_speaker_embedding(
    audio: torch.Tensor | np.ndarray,
    *,
    sample_rate: int,
    model_id: str,
    device: str,
) -> np.ndarray:
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Speaker similarity requires transformers. Install it with: pip install transformers")

    from transformers import AutoFeatureExtractor, WavLMForXVector

    waveform = torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu().numpy()
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = np.clip(waveform / max(peak, 1e-6), -1.0, 1.0)
    if sample_rate != SPEAKER_EMBEDDING_SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=SPEAKER_EMBEDDING_SAMPLE_RATE,
        ).astype(np.float32)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id).eval().to(device)
    inputs = feature_extractor(
        waveform.astype(np.float32),
        sampling_rate=SPEAKER_EMBEDDING_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        embedding = model(**inputs).embeddings
        embedding = F.normalize(embedding, dim=-1)
    return embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _cosine_similarity(reference_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
    reference = np.asarray(reference_embedding, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_embedding, dtype=np.float32).reshape(-1)
    if reference.size == 0 or candidate.size == 0:
        raise ValueError("Speaker embeddings must be non-empty.")
    if reference.shape != candidate.shape:
        raise ValueError(f"Speaker embeddings must have the same shape. Got {reference.shape} and {candidate.shape}.")

    denom = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if denom == 0:
        raise ValueError("Speaker embeddings must have non-zero norm.")
    return float(np.dot(reference, candidate) / denom)


def compute_speaker_similarity(
    reference_audio: torch.Tensor | np.ndarray,
    candidate_audio: torch.Tensor | np.ndarray,
    *,
    reference_sample_rate: int = SPEAKER_EMBEDDING_SAMPLE_RATE,
    candidate_sample_rate: int = SPEAKER_EMBEDDING_SAMPLE_RATE,
    model_id: str = DEFAULT_SPEAKER_ENCODER,
    device: str = "cpu",
) -> float:
    reference_embedding = _compute_speaker_embedding(
        reference_audio,
        sample_rate=reference_sample_rate,
        model_id=model_id,
        device=device,
    )
    candidate_embedding = _compute_speaker_embedding(
        candidate_audio,
        sample_rate=candidate_sample_rate,
        model_id=model_id,
        device=device,
    )
    return _cosine_similarity(reference_embedding, candidate_embedding)
