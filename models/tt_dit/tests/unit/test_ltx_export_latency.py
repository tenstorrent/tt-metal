# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the mp4 export, a request-latency cost outside the device. No device needed."""

import threading

import av
import numpy as np
import pytest
import torch

from models.tt_dit.utils import video


def _yuv_clip(t=9, h=128, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for i in range(t):
        y = (40 + 150 * xx / w + 20 * np.sin(yy / 7.0 + i)).astype(np.uint8)
        u = np.full((h // 2, w // 2), 110 + i, np.uint8)
        v = rng.integers(120, 136, (h // 2, w // 2), dtype=np.uint8)
        frames.append(np.concatenate([y, u.reshape(h // 4, w), v.reshape(h // 4, w)]))
    return np.stack(frames)


def _decode(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray() for f in c.decode(video=0)])


def _psnr(a, b):
    return 10 * np.log10(255.0**2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def test_default_x264_options_favour_latency(monkeypatch):
    for k in ("LTX_EXPORT_PRESET", "LTX_EXPORT_CRF", "LTX_EXPORT_LOSSLESS"):
        monkeypatch.delenv(k, raising=False)
    assert video._x264_options() == {"preset": "ultrafast", "crf": "20"}
    monkeypatch.setenv("LTX_EXPORT_PRESET", "veryfast")
    assert video._x264_options() == {"preset": "veryfast", "crf": "23"}
    monkeypatch.setenv("LTX_EXPORT_CRF", "18")
    assert video._x264_options() == {"preset": "veryfast", "crf": "18"}
    monkeypatch.setenv("LTX_EXPORT_LOSSLESS", "1")
    assert video._x264_options() == {"preset": "veryfast", "qp": "0"}


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_yuv_export_round_trip(tmp_path, monkeypatch, layout):
    for k in ("LTX_EXPORT_PRESET", "LTX_EXPORT_CRF", "LTX_EXPORT_LOSSLESS"):
        monkeypatch.delenv(k, raising=False)
    clip = _yuv_clip()
    if layout == "strided":  # frames that are views into a wider buffer, as a row-padded readback would give
        wide = np.zeros(clip.shape[:2] + (clip.shape[2] + 64,), np.uint8)
        wide[:, :, : clip.shape[2]] = clip
        src = wide[:, :, : clip.shape[2]]
    else:
        src = clip
    audio = video.Audio(waveform=torch.zeros(2, 48000 * clip.shape[0] // 24).uniform_(-0.1, 0.1), sampling_rate=48000)
    out = str(tmp_path / "clip.mp4")
    video.export_video_audio_yuv(src, out, fps=24, audio=audio)

    decoded = _decode(out)
    assert decoded.shape == clip.shape
    assert _psnr(decoded, clip) > 40.0
    with av.open(out) as c:
        assert len(c.streams.audio) == 1


def test_yuv_export_encodes_audio_alongside_video(tmp_path, monkeypatch):
    audio_started = threading.Event()
    encode_audio = video._encode_audio
    encode_video = video.YuvVideoExport._encode_video

    def spy(stream, audio):
        audio_started.set()
        return encode_audio(stream, audio)

    overlapped = []

    def gated(self, stream, yuv_planar):
        # Holds the video track until the audio encode starts: a serial export would never get there.
        overlapped.append(audio_started.wait(timeout=10))
        encode_video(self, stream, yuv_planar)

    monkeypatch.setattr(video, "_encode_audio", spy)
    monkeypatch.setattr(video.YuvVideoExport, "_encode_video", gated)
    clip = _yuv_clip(t=24)
    audio = video.Audio(waveform=torch.zeros(2, 48000).uniform_(-0.1, 0.1), sampling_rate=48000)
    out = str(tmp_path / "clip.mp4")
    video.export_video_audio_yuv(clip, out, fps=24, audio=audio)

    assert overlapped == [True], "audio encode did not start while the video track was encoding"
    with av.open(out) as c:
        samples = sum(f.samples for f in c.decode(audio=0))
    assert abs(samples - 48000) <= 2048  # AAC priming/padding only
