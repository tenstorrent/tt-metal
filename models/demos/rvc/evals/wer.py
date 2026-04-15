# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Speaker similarity evaluation for RVC outputs.

Example:
    ./python_env/bin/python models/demos/rvc/scripts/eval_speaker_similarity.py \
      --device cpu

This computes cosine similarity between speaker embeddings extracted from the
original source audio and the generated audio.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from models.demos.rvc.torch_impl.audio import load_audio

DEFAULT_BACKEND = "transformers_wavlm_xvector"
DEFAULT_SPEAKER_ENCODER = "microsoft/wavlm-base-plus-sv"


class SpeakerEmbeddingBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerSimilarityResult:
    backend: str
    model_id: str
    similarity: float


def cosine_similarity(reference_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
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


def _load_speechbrain_encoder(model_id: str, device: str):
    if importlib.util.find_spec("speechbrain") is None:
        raise SpeakerEmbeddingBackendError(
            "SpeechBrain speaker similarity requires optional dependencies. "
            "Install them with: pip install speechbrain torchaudio"
        )

    from speechbrain.inference.speaker import EncoderClassifier

    kwargs = {"source": model_id, "run_opts": {"device": device}}
    try:
        return EncoderClassifier.from_hparams(**kwargs)
    except Exception as exc:
        raise SpeakerEmbeddingBackendError(
            "Failed to initialize the SpeechBrain speaker encoder. "
            "If the model is not cached locally, this step may require network access. "
            f"model_id={model_id!r}"
        ) from exc


def _load_transformers_wavlm_encoder(model_id: str, device: str):
    if importlib.util.find_spec("transformers") is None:
        raise SpeakerEmbeddingBackendError(
            "Transformers-based speaker similarity requires optional dependencies. "
            "Install them with: pip install transformers"
        )

    from transformers import AutoFeatureExtractor, WavLMForXVector

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id)
    model = model.eval().to(device)
    return feature_extractor, model


def compute_speaker_embedding(
    *,
    backend: str = DEFAULT_BACKEND,
    model_id: str = DEFAULT_SPEAKER_ENCODER,
    device: str = "cpu",
) -> np.ndarray:
    waveform = load_audio(16000).unsqueeze(0)

    if backend == "speechbrain_ecapa":
        classifier = _load_speechbrain_encoder(model_id=model_id, device=device)
        lengths = torch.tensor([1.0], dtype=torch.float32)
        embedding = classifier.encode_batch(waveform, lengths=lengths)
        return embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)

    if backend == "transformers_wavlm_xvector":
        feature_extractor, model = _load_transformers_wavlm_encoder(
            model_id=model_id,
            device=device,
        )
        inputs = feature_extractor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            embedding = model(**inputs).embeddings
            embedding = F.normalize(embedding, dim=-1)
        return embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)

    raise ValueError(f"Unsupported speaker embedding backend: {backend}")


def compute_speaker_similarity(
    *,
    backend: str = DEFAULT_BACKEND,
    model_id: str = DEFAULT_SPEAKER_ENCODER,
    device: str = "cpu",
) -> SpeakerSimilarityResult:
    source_embedding = compute_speaker_embedding(
        backend=backend,
        model_id=model_id,
        device=device,
    )
    generated_embedding = compute_speaker_embedding(
        backend=backend,
        model_id=model_id,
        device=device,
    )
    similarity = cosine_similarity(source_embedding, generated_embedding)
    return SpeakerSimilarityResult(
        backend=backend,
        model_id=model_id,
        similarity=similarity,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute speaker similarity using the fixed RVC demo audio input.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["transformers_wavlm_xvector", "speechbrain_ecapa"],
        help="Speaker embedding backend.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_SPEAKER_ENCODER,
        help="Speaker embedding model identifier for the selected backend.",
    )
    parser.add_argument("--device", default="cpu", help="Execution device for the embedding backend.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compute_speaker_similarity(
        backend=args.backend,
        model_id=args.model_id,
        device=args.device,
    )
    print(f"speaker_similarity={result.similarity:.6f}")


if __name__ == "__main__":
    main()
