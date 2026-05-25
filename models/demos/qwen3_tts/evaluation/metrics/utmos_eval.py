# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
UTMOS evaluation for TTS naturalness (predicted MOS).
Uses the UTMOS22 strong predictor model.
"""

import numpy as np
import torch


def compute_utmos_batch(results, utmos_config):
    """
    Compute UTMOS naturalness scores for a batch of TTS results.

    Args:
        results: List of dicts with 'id', 'wav', 'sr' keys
        utmos_config: Dict with 'model' key

    Returns:
        Dict mapping sample_id -> UTMOS score
    """
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True,
    )
    predictor.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = predictor.to(device)

    utmos_scores = {}
    for r in results:
        wav = r["wav"].astype(np.float32)
        sr = r["sr"]

        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000

        if len(wav) > sr * 10:
            wav = wav[: sr * 10]

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)

        with torch.no_grad():
            score = predictor(wav_tensor, sr)

        utmos_scores[r["id"]] = float(score.item())

    return utmos_scores
