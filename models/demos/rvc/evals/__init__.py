# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from .speaker_similarity import compute_speaker_similarity
from .token_accuracy import (
    DEFAULT_TOKEN_ACCURACY_THRESHOLD,
    TokenAccuracyResult,
    compute_token_accuracy,
    edit_distance,
)
from .wer import (
    ASRBackendError,
    WERResult,
    compute_wer,
    normalize_transcript,
    transcribe_audio,
)

DEFAULT_WHISPER_MODEL = "openai/whisper-medium"


def count_whisper_token(
    audio,
    sample_rate: int,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    whisper_device: str = "cpu",
) -> tuple[str, int]:
    import numpy as np
    import torch
    from transformers import AutoProcessor, pipeline as hf_pipeline

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu()
        audio_wave = audio.to(torch.float32).numpy()
    else:
        audio_wave = np.asarray(audio, dtype=np.float32)

    audio_wave = np.asarray(audio_wave, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio_wave))) if audio_wave.size else 0.0
    if peak > 1.0:
        audio_wave = np.clip(audio_wave / max(peak, 1e-6), -1.0, 1.0)

    if whisper_device == "cpu":
        pipeline_device = -1
        torch_dtype = torch.float32
    else:
        pipeline_device = whisper_device
        torch_dtype = torch.float16

    whisper_asr = hf_pipeline(
        task="automatic-speech-recognition",
        model=whisper_model,
        device=pipeline_device,
        torch_dtype=torch_dtype,
    )
    whisper_processor = AutoProcessor.from_pretrained(whisper_model)

    transcript_result = whisper_asr(
        {"array": audio_wave.astype(np.float32), "sampling_rate": sample_rate},
        return_timestamps=True,
    )
    transcript = transcript_result["text"].strip()
    token_ids = whisper_processor.tokenizer(transcript, add_special_tokens=False).input_ids
    return transcript, len(token_ids)


__all__ = [
    "ASRBackendError",
    "DEFAULT_TOKEN_ACCURACY_THRESHOLD",
    "DEFAULT_WHISPER_MODEL",
    "TokenAccuracyResult",
    "WERResult",
    "compute_speaker_similarity",
    "compute_token_accuracy",
    "compute_wer",
    "count_whisper_token",
    "edit_distance",
    "normalize_transcript",
    "transcribe_audio",
]
