# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from PIL import Image

from models.tt_dit.utils import video_quality
from models.tt_dit.utils.video import Audio, export_to_video, export_video_audio

HEIGHT, WIDTH, FRAMES = 64, 96, 50


@pytest.fixture(scope="module", autouse=True)
def needs_ffmpeg():
    try:
        video_quality.ffmpeg_exe()
    except RuntimeError as error:
        pytest.skip(str(error))


def numbered_frames():
    """Flat frames whose brightness encodes the frame index, so a decoded frame names its index."""
    levels = np.linspace(16, 235, FRAMES).astype(np.uint8)
    return np.broadcast_to(levels[:, None, None, None], (FRAMES, HEIGHT, WIDTH, 3)).copy()


def frame_indices(decoded):
    levels = np.linspace(16, 235, FRAMES)
    means = decoded.reshape(len(decoded), -1).mean(axis=1)
    return [int(np.abs(levels - m).argmin()) for m in means]


@pytest.fixture(scope="module", params=["video", "video_audio"])
def clip(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clip") / f"{request.param}.mp4")
    frames = numbered_frames()
    if request.param == "video":
        export_to_video(frames, path, fps=16, crf=0)
    else:
        # Muxed AAC writes an mp4 edit list, which must not cost frames at the edges.
        waveform = torch.zeros(2, 48000 * FRAMES // 24)
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
        export_video_audio(pixels, path, fps=24, audio=Audio(waveform, 48000))
    return path


def test_decoded_frames_samples_evenly_by_index(clip):
    expected = list(np.linspace(0, FRAMES - 1, 8, dtype=int))

    luma = video_quality.decoded_frames(clip, 8, height=HEIGHT, width=WIDTH)
    assert luma.shape == (8, HEIGHT, WIDTH) and luma.dtype == np.uint8
    assert frame_indices(luma) == expected

    rgb = video_quality.decoded_frames(clip, 8, height=HEIGHT, width=WIDTH, luma=False)
    assert rgb.shape == (8, HEIGHT, WIDTH, 3)
    assert frame_indices(rgb) == expected


def test_decoded_frames_returns_all_frames_of_a_shorter_clip(clip):
    assert len(video_quality.decoded_frames(clip, FRAMES + 10, height=HEIGHT, width=WIDTH)) == FRAMES


def test_decoded_frames_rejects_a_wrong_size(clip, expect_error):
    with expect_error(AssertionError, "wrong height/width"):
        video_quality.decoded_frames(clip, 2, height=HEIGHT, width=WIDTH * 2)


def test_first_frame_pcc_center_crops_a_wider_image():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    # The frame centered between two noise bands: only a center crop lines them up.
    padding = rng.integers(0, 256, (HEIGHT, WIDTH // 2, 3), dtype=np.uint8)
    image = Image.fromarray(np.concatenate([padding, frame, padding], axis=1))

    assert video_quality.first_frame_pcc(frame, image) == pytest.approx(1.0)
    assert video_quality.first_frame_pcc(np.asarray(Image.fromarray(frame).convert("L")), image, luma=True) > 0.99
    assert abs(video_quality.first_frame_pcc(frame, image.transpose(Image.FLIP_LEFT_RIGHT))) < 0.1


def test_assert_first_frame_matches_fails_on_unrelated_image(expect_error):
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    unrelated = Image.fromarray(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    with expect_error(AssertionError, "image-conditioning path"):
        video_quality.assert_first_frame_matches(frame, unrelated, pcc_floor=0.3)
