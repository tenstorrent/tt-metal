# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the mp4 export on the request path. No device needed."""

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


def _pixels_clip(t=9, h=128, w=192):
    """(B, C, F, H, W) float in [-1, 1], as ``decode_latents`` returns it."""
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    frames = [torch.stack([xx / w, yy / h, torch.full_like(xx / w, i / t)]) for i in range(t)]
    return (torch.stack(frames, dim=1) * 1.6 - 0.8)[None]


def _decode(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray() for f in c.decode(video=0)])


def _psnr(a, b):
    return 10 * np.log10(255.0**2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _audio(seconds):
    return video.Audio(waveform=torch.zeros(2, int(48000 * seconds)).uniform_(-0.1, 0.1), sampling_rate=48000)


def test_default_x264_options_favour_latency():
    assert video.X264_OPTIONS == {"preset": "ultrafast", "crf": "20"}


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_yuv_export_round_trip(tmp_path, layout):
    clip = _yuv_clip()
    if layout == "strided":  # frames that are views into a wider buffer, as a row-padded readback would give
        wide = np.zeros(clip.shape[:2] + (clip.shape[2] + 64,), np.uint8)
        wide[:, :, : clip.shape[2]] = clip
        src = wide[:, :, : clip.shape[2]]
    else:
        src = clip
    out = str(tmp_path / "clip.mp4")
    video.export_video_audio_yuv(src, out, fps=24, audio=_audio(clip.shape[0] / 24))

    decoded = _decode(out)
    assert decoded.shape == clip.shape
    assert _psnr(decoded, clip) > 40.0
    with av.open(out) as c:
        assert len(c.streams.audio) == 1


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
