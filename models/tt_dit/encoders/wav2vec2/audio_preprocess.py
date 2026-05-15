# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import Wav2Vec2Processor


WAV2VEC2_SAMPLE_RATE = 16000
WAV2VEC2_HZ = 50  # wav2vec2 outputs hidden states at 50 Hz (16 kHz / 320 stride).
S2V_VIDEO_RATE = 30  # reference WAN 2.2 S2V resamples wav2vec2 features to 30 Hz before bucketing.


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


def linear_interpolation(
    features: torch.Tensor, *, input_fps: int, output_fps: int, output_len: int | None = None
) -> torch.Tensor:
    """1-D linear interpolation along the time axis.

    Mirrors ``wan/modules/s2v/audio_encoder.py:linear_interpolation``. The
    reference resamples wav2vec2 features from 50 Hz to ``video_rate`` (30 Hz
    for the production S2V config) before bucketing — skipping this step
    leaves the bucket indices misaligned with the model's expected per-frame
    rate.

    Args:
        features: ``[T, C]`` float features at ``input_fps``.
        input_fps: Source frame rate of ``features``.
        output_fps: Target frame rate.
        output_len: Optional explicit output length; otherwise computed as
            ``int(T / input_fps * output_fps)``.

    Returns:
        ``[output_len, C]`` features at ``output_fps``.
    """
    assert features.dim() == 2, f"expected [T, C] got {tuple(features.shape)}"
    features_1CT = features.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    if output_len is None:
        output_len = int(features_1CT.shape[2] / float(input_fps) * output_fps)
    resampled = torch.nn.functional.interpolate(features_1CT, size=output_len, align_corners=True, mode="linear")
    return resampled.squeeze(0).transpose(0, 1).contiguous()  # [output_len, C]


def _get_sample_indices(*, original_fps: int, total_frames: int, target_fps: int, num_sample: int) -> torch.Tensor:
    """Evenly-spaced indices into a ``total_frames`` sequence at ``original_fps``,
    sampling at ``target_fps``.

    Mirrors ``wan/modules/s2v/audio_encoder.py:get_sample_indices`` with
    ``fixed_start=0`` (single-clip pipeline always starts at the audio
    origin).
    """
    required_duration = num_sample / target_fps
    if required_duration > total_frames / original_fps:
        msg = f"required_duration={required_duration}s exceeds audio length {total_frames / original_fps}s"
        raise ValueError(msg)
    # endpoint=False linspace over [0, required_duration).
    step = required_duration / num_sample
    time_points = torch.arange(num_sample, dtype=torch.float64) * step
    frame_indices = torch.round(time_points * original_fps).to(torch.long)
    return torch.clamp(frame_indices, 0, total_frames - 1)


def get_audio_embed_bucket_fps(
    audio_embed: torch.Tensor,
    *,
    fps: int = 16,
    batch_frames: int = 81,
    video_rate: int = S2V_VIDEO_RATE,
) -> tuple[torch.Tensor, int]:
    """Bucket wav2vec2 features (resampled to ``video_rate``) to ``fps`` video frames.

    Mirrors ``wan/modules/s2v/audio_encoder.py:get_audio_embed_bucket_fps``
    for the single-clip path with ``m=0`` (one audio feature per video
    frame). The caller is expected to have already resampled the wav2vec2
    output from 50 Hz to ``video_rate`` Hz via :func:`linear_interpolation`.

    Args:
        audio_embed: ``[T_audio, C]`` float features at ``video_rate``.
        fps: Target video frame rate.
        batch_frames: Number of video frames in one clip.
        video_rate: Frame rate that ``audio_embed`` was resampled to.

    Returns:
        Tuple ``(aligned, num_clips)`` where ``aligned`` has shape
        ``[num_clips * batch_frames, C]``. For single-clip audio
        ``num_clips == 1`` and the output is ``[batch_frames, C]``.
    """
    assert audio_embed.dim() == 2, f"expected [T, C] got {tuple(audio_embed.shape)}"
    audio_frame_num, audio_dim = audio_embed.shape
    scale = video_rate / fps  # float

    # min_batch_num is the number of full clips fitting in the audio
    # (the reference always rounds up by +1 to guarantee enough samples).
    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
    bucket_num = min_batch_num * batch_frames

    # The reference's sample-index range extends past the real audio to
    # accommodate the +1 rounding above; indices past the audio length
    # produce zero embeddings (the model treats those as "no audio for
    # this video frame"). See ``wan/modules/s2v/audio_encoder.py:163-184``.
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    total_frames = audio_frame_num + padd_audio_num

    frame_indices = _get_sample_indices(
        original_fps=video_rate,
        total_frames=total_frames,
        target_fps=fps,
        num_sample=bucket_num,
    )
    in_range = frame_indices < audio_frame_num
    safe_indices = torch.where(in_range, frame_indices, torch.zeros_like(frame_indices))
    aligned = audio_embed[safe_indices]
    aligned = aligned * in_range.unsqueeze(-1).to(aligned.dtype)
    return aligned, min_batch_num
