# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the streamed temporal assembly: the frames handed out clip by clip, joined,
are exactly the batch assembly's trimmed video, and the returned buffer is the batch result."""

import threading

import numpy as np
import pytest

from ....models.vae.minimax_h3.temporal_minimax_h3 import (
    StreamingClipAssembly,
    assemble_clip_parts,
    blend_clip_frames,
    split_clip,
)

# The served config: 5-token chunks, 4x temporal, 3 tokens dropped, 3 pre-padding frames, 5-frame cross-fade.
CHUNK_FRAMES = 20
PRE_PADDING = 3
OVERLAP = 5


def _clips(num_clips: int, token_drop: int, seed: int = 0) -> list[np.ndarray]:
    frames = CHUNK_FRAMES * (2 if token_drop else 1)
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(frames, 12, 8), dtype=np.uint8) for _ in range(num_clips)]


def _batch(clips: list[np.ndarray], token_drop: int, trim: int) -> np.ndarray:
    parts, overlap = [], None
    for clip in clips:
        part, tail = split_clip(clip, CHUNK_FRAMES, PRE_PADDING, token_drop)
        parts.append((part, overlap))
        overlap = tail
    if overlap is not None:
        parts.append((overlap, None))
    result = assemble_clip_parts(parts, OVERLAP)
    return result[: result.shape[0] - trim]


def _stream(clips, token_drop, trim, on_frames):
    stream = StreamingClipAssembly(
        len(clips),
        chunk_num_frames=CHUNK_FRAMES,
        frame_pre_padding=PRE_PADDING,
        token_drop=token_drop,
        frame_overlap=OVERLAP,
        trim=trim,
        on_frames=on_frames,
    )
    try:
        for clip in clips:
            stream.add(clip)
        return stream.result()
    finally:
        stream.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("num_clips", [6, 9, 14])
def test_assembled_frames_are_the_whole_seam_blend(num_clips, dtype):
    # uint8 seams go frame by frame through scratch frames; `blend_clip_frames` on the whole seam is the reference.
    rng = np.random.default_rng(num_clips)
    parts, overlap = [], None
    for _ in range(num_clips):  # a served clip: 28 frames, a 17-frame part and a 5-frame tail
        clip = rng.integers(0, 256, size=(28, 6, 4)).astype(dtype)
        part, tail = split_clip(clip, CHUNK_FRAMES, PRE_PADDING, 3)
        parts.append((part, overlap))
        overlap = tail
    parts[-1] = (parts[-1][0][:3], parts[-1][1])  # a last chunk shorter than the cross-fade
    parts.append((overlap, None))
    expected = np.concatenate(
        [segment if previous is None else blend_clip_frames(previous, segment, OVERLAP) for segment, previous in parts]
    )
    result = assemble_clip_parts(parts, OVERLAP)
    assert result.dtype == expected.dtype and np.array_equal(result, expected)


@pytest.mark.parametrize("num_clips", [1, 2, 15])
@pytest.mark.parametrize("token_drop", [0, 3])
@pytest.mark.parametrize("trim", [0, 3, 7])
def test_streamed_frames_are_the_batch_assembly(num_clips, token_drop, trim):
    clips = _clips(num_clips, token_drop)
    expected = _batch(clips, token_drop, trim)
    runs = []
    full = _stream(clips, token_drop, trim, lambda frames: runs.append(frames.copy()))

    assert np.array_equal(np.concatenate(runs), expected)
    assert np.array_equal(full[: full.shape[0] - trim], expected)
    # A run goes out per clip (the last clip's run carries the tail); none is empty.
    assert len(runs) == num_clips and all(len(run) for run in runs)


def test_each_run_goes_out_when_its_clip_lands():
    clips = _clips(4, 3)
    handed = []
    seen = threading.Event()

    def on_frames(frames):
        assert not frames.flags.writeable
        handed.append(len(frames))
        seen.set()

    stream = StreamingClipAssembly(
        len(clips),
        chunk_num_frames=CHUNK_FRAMES,
        frame_pre_padding=PRE_PADDING,
        token_drop=3,
        frame_overlap=OVERLAP,
        trim=0,
        on_frames=on_frames,
    )
    try:
        stream.add(clips[0])
        # The first clip's frames are final before any later clip exists.
        assert seen.wait(5)
        assert handed == [CHUNK_FRAMES - PRE_PADDING]
        for clip in clips[1:]:
            stream.add(clip)
        stream.result()
    finally:
        stream.close()
    assert handed == [17, 17, 17, 34]


def test_a_failing_consumer_fails_the_decode_and_stops_writing():
    calls = []

    def on_frames(frames):
        calls.append(len(frames))
        raise RuntimeError("encoder gone")

    with pytest.raises(RuntimeError, match="encoder gone"):
        _stream(_clips(3, 3), 3, 0, on_frames)
    assert calls == [17]


def test_a_missing_clip_fails_the_result():
    stream = StreamingClipAssembly(
        3,
        chunk_num_frames=CHUNK_FRAMES,
        frame_pre_padding=PRE_PADDING,
        token_drop=3,
        frame_overlap=OVERLAP,
        trim=0,
        on_frames=lambda frames: None,
    )
    try:
        for clip in _clips(2, 3):
            stream.add(clip)
        with pytest.raises(ValueError, match="2 clips for a 3-clip decode"):
            stream.result()
    finally:
        stream.close()


def test_a_clip_after_a_consumer_error_raises_it():
    failed = threading.Event()

    def on_frames(frames):
        failed.set()
        raise RuntimeError("encoder gone")

    stream = StreamingClipAssembly(
        3,
        chunk_num_frames=CHUNK_FRAMES,
        frame_pre_padding=PRE_PADDING,
        token_drop=3,
        frame_overlap=OVERLAP,
        trim=0,
        on_frames=on_frames,
    )
    clips = _clips(3, 3)
    try:
        stream.add(clips[0])
        assert failed.wait(5)
        stream._futures[0].exception(timeout=5)
        with pytest.raises(RuntimeError, match="encoder gone"):
            stream.add(clips[1])
        assert len(stream._futures) == 1
    finally:
        stream.close()
