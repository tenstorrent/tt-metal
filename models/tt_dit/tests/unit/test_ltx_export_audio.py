# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the mp4 export encodes audio off the main thread and still muxes every stream."""

import threading

import av
import numpy as np
import pytest
import torch

from models.tt_dit.utils import video


@pytest.mark.parametrize("yuv", [True, False], ids=["yuv", "pixels"])
def test_export_encodes_audio_alongside_video(tmp_path, monkeypatch, yuv):
    threads = []
    encode_audio = video._encode_audio

    def spy(stream, audio):
        threads.append(threading.current_thread())
        return encode_audio(stream, audio)

    monkeypatch.setattr(video, "_encode_audio", spy)
    out = str(tmp_path / "clip.mp4")
    audio = video.Audio(waveform=torch.zeros(2, 48000).uniform_(-0.1, 0.1), sampling_rate=48000)
    if yuv:  # (T, H * 3 / 2, W) planar yuv420p frames
        frames = np.random.default_rng(0).integers(16, 236, (24, 192, 192), dtype=np.uint8)
        video.export_video_audio_yuv(frames, out, fps=24, audio=audio)
    else:  # (B, C, F, H, W) float in [-1, 1], as decode_latents returns it
        video.export_video_audio(torch.rand(1, 3, 24, 128, 192) * 2 - 1, out, fps=24, audio=audio)

    assert threads and threads[0] is not threading.main_thread()
    with av.open(out) as c:
        assert sum(1 for _ in c.decode(video=0)) == 24
    with av.open(out) as c:
        samples = sum(f.samples for f in c.decode(audio=0))
    assert abs(samples - 48000) <= 2048  # AAC priming/padding only
