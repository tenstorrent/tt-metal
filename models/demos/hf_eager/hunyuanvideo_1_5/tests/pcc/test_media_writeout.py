# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract for generated-frame writeout. No device is opened."""

import os

import numpy as np
import pytest
from PIL import Image

from models.demos.hf_eager.hunyuanvideo_1_5.tt.media_writeout import save_generated_frames, write_png_frames


def _frames(count=6, height=48, width=80):
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)) for _ in range(count)]


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


def test_threaded_png_output_is_byte_identical_to_serial(tmp_path):
    frames = _frames()
    serial, threaded = tmp_path / "serial", tmp_path / "threaded"
    serial.mkdir()
    threaded.mkdir()

    write_png_frames(frames, str(serial), threaded=False)
    write_png_frames(frames, str(threaded), threaded=True)

    names = sorted(os.listdir(serial))
    assert names == sorted(os.listdir(threaded)) and len(names) == len(frames)
    for name in names:
        assert _read(serial / name) == _read(threaded / name)


@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_every_compression_level_round_trips_the_exact_pixels(tmp_path, level):
    """PNG is lossless, so the level is a size/time knob and never a quality one."""
    frames = _frames()
    outdir = tmp_path / f"level{level}"
    outdir.mkdir()
    paths = write_png_frames(frames, str(outdir), compress_level=level, threaded=True)
    for frame, path in zip(frames, paths):
        assert np.array_equal(np.asarray(Image.open(path).convert("RGB")), np.asarray(frame))


def test_the_default_configuration_reproduces_the_previous_inline_writeout(tmp_path):
    frames = _frames()
    reference = tmp_path / "reference"
    reference.mkdir()
    for index, frame in enumerate(frames):
        frame.save(reference / f"frame_{index:03d}.png")
    frames[0].save(reference / "tt_blackhole.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)

    outdir = tmp_path / "module"
    timings = save_generated_frames(frames, str(outdir), write_mp4=False)

    assert timings["fast"] is False and timings["png_compress_level"] == 6
    for index in range(len(frames)):
        name = f"frame_{index:03d}.png"
        assert _read(outdir / name) == _read(reference / name)
    assert _read(outdir / "tt_blackhole.gif") == _read(reference / "tt_blackhole.gif")


def test_the_fast_gate_changes_only_wall_time_not_bytes(tmp_path):
    frames = _frames()
    slow = tmp_path / "slow"
    fast = tmp_path / "fast"
    save_generated_frames(frames, str(slow), fast=False, write_gif=False, write_mp4=False)
    save_generated_frames(frames, str(fast), fast=True, write_gif=False, write_mp4=False)
    for index in range(len(frames)):
        name = f"frame_{index:03d}.png"
        assert _read(slow / name) == _read(fast / name)


def test_env_gates_default_to_todays_behaviour(tmp_path, monkeypatch):
    monkeypatch.delenv("HY_FAST_WRITEOUT", raising=False)
    monkeypatch.delenv("HY_PNG_COMPRESS", raising=False)
    monkeypatch.delenv("HY_SAVE_GIF", raising=False)
    timings = save_generated_frames(_frames(2), str(tmp_path / "out"), write_mp4=False)
    assert timings["fast"] is False
    assert timings["png_compress_level"] == 6
    assert timings["gif"] is True


def test_env_gates_are_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("HY_FAST_WRITEOUT", "1")
    monkeypatch.setenv("HY_PNG_COMPRESS", "1")
    monkeypatch.setenv("HY_SAVE_GIF", "0")
    outdir = tmp_path / "out"
    timings = save_generated_frames(_frames(2), str(outdir), write_mp4=False)
    assert timings["fast"] is True and timings["png_compress_level"] == 1 and timings["gif"] is False
    assert not (outdir / "tt_blackhole.gif").exists()


def test_an_illegal_compression_level_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HY_PNG_COMPRESS", "12")
    with pytest.raises(ValueError, match="0..9"):
        save_generated_frames(_frames(2), str(tmp_path / "out"), write_mp4=False)


def test_float_arrays_are_converted_the_same_way_the_test_harness_did(tmp_path):
    """The generation path hands over float arrays in [0, 1], not PIL images."""
    rng = np.random.default_rng(1)
    arrays = [rng.random((16, 24, 3)) for _ in range(3)]
    outdir = tmp_path / "out"
    save_generated_frames(arrays, str(outdir), write_gif=False, write_mp4=False)
    for index, array in enumerate(arrays):
        expected = (array.clip(0, 1) * 255).astype("uint8")
        written = np.asarray(Image.open(outdir / f"frame_{index:03d}.png").convert("RGB"))
        assert np.array_equal(written, expected)
