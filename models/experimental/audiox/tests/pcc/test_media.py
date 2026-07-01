# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from models.experimental.audiox.demo.media import prepare_audio_prompt, prepare_video_prompt, resample_output_audio


def test_prepare_video_prompt_resamples_and_resizes():
    frames = torch.arange(3 * 3 * 8 * 12, dtype=torch.float32).reshape(3, 3, 8, 12)
    prompt = prepare_video_prompt(frames, target_frames=5, image_size=16)
    assert prompt.shape == (1, 5, 3, 16, 16)


def test_prepare_video_prompt_repeats_single_frame():
    frame = torch.ones(1, 3, 10, 10)
    prompt = prepare_video_prompt(frame, target_frames=4, image_size=10)
    assert prompt.shape == (1, 4, 3, 10, 10)
    assert torch.all(prompt == 1.0)


def test_prepare_video_prompt_rejects_non_rgb_frames():
    with pytest.raises(ValueError, match="expected RGB frames"):
        prepare_video_prompt(torch.ones(2, 1, 10, 10), target_frames=4, image_size=10)


def test_prepare_video_prompt_rejects_bad_rank():
    with pytest.raises(ValueError, match=r"expected \[T, C, H, W\] frames"):
        prepare_video_prompt(torch.ones(3, 10, 10), target_frames=4, image_size=10)


def test_prepare_audio_prompt_repeats_mono_and_pads():
    waveform = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    prompt = prepare_audio_prompt(waveform, target_samples=8, target_channels=2)
    assert prompt.shape == (1, 2, 8)
    assert torch.all(prompt[0, 0, :6] == waveform[0])
    assert torch.all(prompt[0, 1, :6] == waveform[0])
    assert torch.all(prompt[:, :, 6:] == 0)


def test_prepare_audio_prompt_truncates_extra_samples():
    waveform = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    prompt = prepare_audio_prompt(waveform, target_samples=6, target_channels=2)
    assert prompt.shape == (1, 2, 6)
    assert torch.all(prompt[0] == waveform[:, :6])


def test_resample_output_audio_changes_length():
    audio = torch.randn(1, 2, 4410)
    out = resample_output_audio(audio, input_sample_rate=44100, output_sample_rate=16000)
    assert out.shape == (1, 2, 1600)


def test_resample_output_audio_identity_when_rates_match():
    audio = torch.randn(1, 2, 32)
    out = resample_output_audio(audio, input_sample_rate=16000, output_sample_rate=16000)
    assert out is audio
