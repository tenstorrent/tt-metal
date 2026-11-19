# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only: the threaded YUV export (LTX_ASYNC_EXPORT=1) takes deferred frames and reports encode errors."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("av")

from models.tt_dit.utils.video import Audio, YuvVideoExport, export_video_audio_yuv
from models.tt_dit.utils.yuv_d2h import DeferredYuvPlanar


@pytest.fixture
def clip():
    g = torch.Generator().manual_seed(0)
    frames, height, width = 17, 64, 96
    yuv = torch.randint(0, 256, (frames, height * 3 // 2, width), generator=g, dtype=torch.uint8).numpy()
    rate = 48000
    audio = Audio(waveform=torch.rand(2, rate * frames // 24, generator=g) * 2 - 1, sampling_rate=rate)
    return yuv, audio


def test_deferred_frames_export_same_bytes(tmp_path, clip):
    yuv, audio = clip
    ref, out = tmp_path / "ref.mp4", tmp_path / "out.mp4"
    export_video_audio_yuv(yuv, str(ref), fps=24, audio=audio)
    deferred = DeferredYuvPlanar(lambda: yuv, yuv.shape)
    YuvVideoExport(deferred, str(out), fps=24, audio_sampling_rate=audio.sampling_rate).finish(audio)
    assert out.read_bytes() == ref.read_bytes()


def test_encode_error_surfaces_on_finish(tmp_path, clip, expect_error):
    yuv, _ = clip
    bad = yuv[:, :, :-1]  # odd width: not a yuv420p frame, so the worker's encode fails
    export = YuvVideoExport(bad, str(tmp_path / "out.mp4"), fps=24)
    with expect_error(ValueError, "Unexpected numpy array shape"):
        export.finish(None)
