# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Token-level transcript accuracy between two audio waveforms."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOKEN_ACCURACY_THRESHOLD = 0.90
DEFAULT_WHISPER_MODEL = "nyrahealth/CrisperWhisper"


@dataclass(frozen=True)
class TokenAccuracyResult:
    reference_transcript: str
    candidate_transcript: str
    reference_num_tokens: int
    candidate_num_tokens: int
    token_edit_distance: int
    token_accuracy: float
    passed: bool


def _audio_to_waveform(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    import numpy as np
    import torch

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu()
        waveform = audio.to(torch.float32).numpy()
    else:
        waveform = np.asarray(audio, dtype=np.float32)

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = np.clip(waveform / max(peak, 1e-6), -1.0, 1.0)
    return waveform


def edit_distance(reference_tokens: list[int], candidate_tokens: list[int]) -> int:
    rows = len(reference_tokens) + 1
    cols = len(candidate_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if reference_tokens[row - 1] == candidate_tokens[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + cost,
            )
    return dp[-1][-1]


def compute_token_accuracy(
    reference_audio: torch.Tensor | np.ndarray,
    candidate_audio: torch.Tensor | np.ndarray,
    *,
    reference_sample_rate: int,
    candidate_sample_rate: int,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    whisper_device: str = "cpu",
    threshold: float = DEFAULT_TOKEN_ACCURACY_THRESHOLD,
) -> TokenAccuracyResult:
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from transformers import pipeline as hf_pipeline

    if whisper_device == "cpu":
        pipeline_device = -1
        torch_dtype = torch.float32
    else:
        pipeline_device = whisper_device
        torch_dtype = torch.float16

    asr = hf_pipeline(
        task="automatic-speech-recognition",
        model=whisper_model,
        device=pipeline_device,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(whisper_model)

    reference_waveform = _audio_to_waveform(reference_audio)
    candidate_waveform = _audio_to_waveform(candidate_audio)

    reference_result = asr(
        {"array": reference_waveform.astype(np.float32), "sampling_rate": reference_sample_rate},
        return_timestamps=True,
    )
    candidate_result = asr(
        {"array": candidate_waveform.astype(np.float32), "sampling_rate": candidate_sample_rate},
        return_timestamps=True,
    )

    reference_transcript = reference_result["text"].strip()
    candidate_transcript = candidate_result["text"].strip()
    reference_tokens = processor.tokenizer(reference_transcript, add_special_tokens=False).input_ids
    candidate_tokens = processor.tokenizer(candidate_transcript, add_special_tokens=False).input_ids

    if not reference_tokens:
        raise ValueError("Reference transcript produced no tokens.")

    distance = edit_distance(reference_tokens, candidate_tokens)
    token_accuracy = max(0.0, 1.0 - distance / len(reference_tokens))
    return TokenAccuracyResult(
        reference_transcript=reference_transcript,
        candidate_transcript=candidate_transcript,
        reference_num_tokens=len(reference_tokens),
        candidate_num_tokens=len(candidate_tokens),
        token_edit_distance=distance,
        token_accuracy=token_accuracy,
        passed=token_accuracy > threshold,
    )
