# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only: the threaded YUV export (LTX_ASYNC_EXPORT=1) writes the same mp4 bytes as the serial one."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("av")

from models.tt_dit.utils.video import Audio, YuvVideoExport, export_video_audio_yuv


@pytest.fixture
def clip():
    g = torch.Generator().manual_seed(0)
    frames, height, width = 17, 64, 96
    yuv = torch.randint(0, 256, (frames, height * 3 // 2, width), generator=g, dtype=torch.uint8).numpy()
    rate = 48000
    audio = Audio(waveform=torch.rand(2, rate * frames // 24, generator=g) * 2 - 1, sampling_rate=rate)
    return yuv, audio


@pytest.mark.parametrize("with_audio", [True, False])
def test_threaded_export_matches_serial(tmp_path, clip, with_audio):
    yuv, audio = clip
    audio = audio if with_audio else None
    ref, out = tmp_path / "ref.mp4", tmp_path / "out.mp4"
    export_video_audio_yuv(yuv, str(ref), fps=24, audio=audio)
    rate = audio.sampling_rate if audio is not None else None
    YuvVideoExport(yuv, str(out), fps=24, audio_sampling_rate=rate).finish(audio)
    assert out.read_bytes() == ref.read_bytes()


def test_audio_rate_mismatch_raises_and_closes(tmp_path, clip, expect_error):
    yuv, audio = clip
    export = YuvVideoExport(yuv, str(tmp_path / "out.mp4"), fps=24, audio_sampling_rate=audio.sampling_rate)
    with expect_error(ValueError, "does not match the declared rate"):
        export.finish(None)
    assert not export._thread.is_alive()


def test_encode_error_surfaces_on_finish(tmp_path, clip, expect_error):
    yuv, _ = clip
    bad = yuv[:, :, :-1]  # odd width: not a yuv420p frame, so the worker's encode fails
    export = YuvVideoExport(bad, str(tmp_path / "out.mp4"), fps=24)
    with expect_error(ValueError, "Unexpected numpy array shape"):
        export.finish(None)
    assert not export._thread.is_alive()


class _Deferred:
    """Stands in for ``DeferredYuvPlanar``: records which thread assembled the frames."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.thread = None

    def result(self):
        import threading

        self.thread = threading.current_thread().name
        return self._array


def test_deferred_frames_export_same_bytes_on_worker(tmp_path, clip):
    yuv, audio = clip
    ref, out = tmp_path / "ref.mp4", tmp_path / "out.mp4"
    export_video_audio_yuv(yuv, str(ref), fps=24, audio=audio)
    deferred = _Deferred(yuv)
    export = YuvVideoExport(deferred, str(out), fps=24, audio_sampling_rate=audio.sampling_rate)
    export.finish(audio)
    assert out.read_bytes() == ref.read_bytes()
    assert deferred.thread == "yuv-video-export"


def test_deferred_yuv_planar_resolves_lazily_and_checks_shape(expect_error):
    DeferredYuvPlanar = pytest.importorskip("models.tt_dit.utils.yuv_d2h").DeferredYuvPlanar

    flat = torch.randint(0, 256, (5, 6 * 4 * 3 // 2), dtype=torch.uint8).numpy()
    calls = []

    def produce():
        calls.append(1)
        return flat

    deferred = DeferredYuvPlanar(lambda: produce().reshape(5, 9, 4), (5, 9, 4))
    assert deferred.shape == (5, 9, 4) and not calls
    assert (deferred.result() == flat.reshape(5, 9, 4)).all() and calls == [1]
    with expect_error(RuntimeError, r"expected \(4, 36\)"):
        DeferredYuvPlanar(produce, (4, 36)).result()
