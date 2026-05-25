# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Speaker Similarity (SIM) evaluation for voice cloning.
Uses WavLM-based speaker verification model.
"""

import numpy as np
import torch


def compute_speaker_embedding(model, feature_extractor, wav, sr, device="cpu"):
    """Extract speaker embedding from audio using WavLM."""
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.embeddings
        embedding = torch.nn.functional.normalize(embedding, dim=-1)

    return embedding.cpu()


def compute_sim_batch(results, ref_results, sim_config):
    """
    Compute speaker similarity between synthesized and reference audio.

    Args:
        results: List of dicts with 'id', 'wav', 'sr' (synthesized audio)
        ref_results: List of dicts with 'id', 'wav', 'sr' (reference audio)
        sim_config: Dict with 'model' key

    Returns:
        Dict mapping sample_id -> cosine similarity score
    """
    from transformers import AutoFeatureExtractor, WavLMForXVector

    model_name = sim_config.get("model", "microsoft/wavlm-base-plus-sv")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    ref_embeddings = {}
    for r in ref_results:
        emb = compute_speaker_embedding(
            model, feature_extractor, r["wav"].astype(np.float32), r["sr"], device
        )
        ref_embeddings[r["id"]] = emb

    sim_scores = {}
    for r in results:
        if r["id"] not in ref_embeddings:
            continue
        syn_emb = compute_speaker_embedding(
            model, feature_extractor, r["wav"].astype(np.float32), r["sr"], device
        )
        ref_emb = ref_embeddings[r["id"]]
        similarity = torch.nn.functional.cosine_similarity(syn_emb, ref_emb).item()
        sim_scores[r["id"]] = float(similarity)

    return sim_scores
