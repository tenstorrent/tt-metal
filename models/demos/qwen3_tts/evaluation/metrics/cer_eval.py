# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CER (Character Error Rate) evaluation for Japanese TTS.
Uses Whisper Large v3 as the ASR backend.
"""

import re

import numpy as np
import torch


def normalize_japanese_text(text):
    """Normalize text for CER comparison: remove punctuation, normalize whitespace."""
    text = re.sub(r"[、。！？「」『』（）\[\]【】…・～〜ー\s　,.!?;:\-\"']", "", text)
    text = text.strip()
    return text


def compute_cer_single(reference, hypothesis):
    """Compute character error rate between reference and hypothesis."""
    ref_chars = list(normalize_japanese_text(reference))
    hyp_chars = list(normalize_japanese_text(hypothesis))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=int)
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def compute_cer_batch(results, asr_config):
    """
    Compute CER for a batch of TTS results using Whisper ASR.

    Args:
        results: List of dicts with 'id', 'text', 'wav', 'sr' keys
        asr_config: Dict with 'asr_model' and 'language' keys

    Returns:
        Dict mapping sample_id -> CER value
    """
    import whisper

    model_name = asr_config.get("asr_model", "openai/whisper-large-v3")
    whisper_size = model_name.split("whisper-")[-1] if "whisper-" in model_name else "large"
    model = whisper.load_model(whisper_size)

    cer_scores = {}
    for r in results:
        audio = r["wav"].astype(np.float32)
        if r["sr"] != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=r["sr"], target_sr=16000)

        result = model.transcribe(audio, language="ja", task="transcribe")
        hypothesis = result["text"]
        cer = compute_cer_single(r["text"], hypothesis)
        cer_scores[r["id"]] = cer

    return cer_scores
