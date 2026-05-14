# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import Wav2Vec2Processor


WAV2VEC2_SAMPLE_RATE = 16000
WAV2VEC2_HZ = 50  # Wav2Vec2-base outputs hidden states at 50 Hz (16 kHz / 320 stride).


def load_audio_to_input_values(
    audio_path: str,
    processor: Wav2Vec2Processor,
    *,
    target_sr: int = WAV2VEC2_SAMPLE_RATE,
) -> torch.Tensor:
    """Load an audio file and run it through the HF Wav2Vec2Processor on CPU.

    Returns a `[1, num_samples]` float32 tensor suitable for upload to device.
    """
    import librosa

    waveform, _sr = librosa.load(audio_path, sr=target_sr, mono=True)
    inputs = processor(waveform, sampling_rate=target_sr, return_tensors="pt")
    return inputs.input_values


def get_audio_embed_bucket_fps(
    audio_embed: torch.Tensor,
    *,
    fps: int = 16,
    batch_frames: int = 81,
    encoder_hz: int = WAV2VEC2_HZ,
) -> tuple[torch.Tensor, int]:
    """Bucket Wav2Vec2 features to align with video frames at `fps`.

    This mirrors the reference WAN 2.2 S2V's `get_audio_embed_bucket_fps` for the
    single-clip case: it picks `batch_frames` evenly spaced samples from the
    `encoder_hz`-rate feature sequence and pads/truncates so the output length
    matches `batch_frames`.

    Args:
        audio_embed: `[T_audio, C]` float tensor of per-step Wav2Vec2 features
            (final hidden state).
        fps: Target video frame rate.
        batch_frames: Number of video frames in one clip.
        encoder_hz: Wav2Vec2 feature rate (50 Hz for base).

    Returns:
        Tuple `(aligned, num_audio_frames_available)` where `aligned` has shape
        `[batch_frames, C]`. `num_audio_frames_available` is the number of
        non-padded frames (useful for masking when audio is shorter than the
        video).
    """
    assert audio_embed.dim() == 2
    t_audio, c = audio_embed.shape
    stride = encoder_hz // fps  # 50 / 16 -> 3 (integer)

    indices = torch.arange(batch_frames) * stride
    valid = indices < t_audio
    num_valid = int(valid.sum().item())

    clamped = torch.clamp(indices, max=t_audio - 1)
    aligned = audio_embed[clamped]
    if num_valid < batch_frames:
        aligned[num_valid:].zero_()
    return aligned, num_valid
