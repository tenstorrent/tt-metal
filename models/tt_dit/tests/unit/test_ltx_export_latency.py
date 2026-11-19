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


def _yuv_clip(t, h=128, w=192):
    """(T, H * 3 / 2, W) planar yuv420p frames."""
    return np.random.default_rng(0).integers(16, 236, (t, h * 3 // 2, w), dtype=np.uint8)


def _pixels_clip(t, h=128, w=192):
    """(B, C, F, H, W) float in [-1, 1], as ``decode_latents`` returns it."""
    return torch.rand(1, 3, t, h, w) * 2 - 1


def _decode(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray() for f in c.decode(video=0)])


def _audio(seconds):
    return video.Audio(waveform=torch.zeros(2, int(48000 * seconds)).uniform_(-0.1, 0.1), sampling_rate=48000)


def _export_yuv(out, audio):
    video.export_video_audio_yuv(_yuv_clip(t=24), out, fps=24, audio=audio)


def _export_pixels(out, audio):
    video.export_video_audio(_pixels_clip(t=24), out, fps=24, audio=audio)


@pytest.mark.parametrize("export", [_export_yuv, _export_pixels], ids=["yuv", "pixels"])
def test_export_encodes_audio_alongside_video(tmp_path, monkeypatch, export):
    threads = []
    encode_audio = video._encode_audio

    def spy(stream, audio):
        threads.append(threading.current_thread())
        return encode_audio(stream, audio)

    monkeypatch.setattr(video, "_encode_audio", spy)
    out = str(tmp_path / "clip.mp4")
    export(out, _audio(1.0))

    assert threads and threads[0] is not threading.main_thread()
    assert _decode(out).shape[0] == 24
    with av.open(out) as c:
        samples = sum(f.samples for f in c.decode(audio=0))
    assert abs(samples - 48000) <= 2048  # AAC priming/padding only


@pytest.mark.parametrize("export", [_export_yuv, _export_pixels], ids=["yuv", "pixels"])
def test_export_without_audio_has_no_audio_stream(tmp_path, export):
    out = str(tmp_path / "clip.mp4")
    export(out, None)
    with av.open(out) as c:
        assert len(c.streams.audio) == 0 and c.streams.video[0].frames in (0, 24)
