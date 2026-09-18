# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
from contextlib import contextmanager
from fractions import Fraction

import pytest

from models.tt_dit.utils import vbench_cpu


def test_early_motion_decision_matches_exhaustive_vbench_count():
    # Upstream check_move returns True on reaching round(frame_count / 4)
    # positive pair scores. Its zero-pair result is False even at a zero target.
    for pairs in range(13):
        required = round((pairs + 1) / 4)
        for scores in (
            itertools.product((0.0, 0.5, 1.0), repeat=pairs)
            if pairs < 6
            else itertools.product((0.0, 1.0), repeat=pairs)
        ):
            expected = bool(scores) and sum(score > 0.5 for score in scores) >= required
            assert vbench_cpu.motion_decision(iter(scores), pairs=pairs, threshold=0.5, required=required) == expected


@pytest.mark.parametrize("score,expected,consumed", [(1.0, True, 12), (0.0, False, 37)])
def test_flow_stops_when_later_pairs_cannot_change_answer(score, expected, consumed):
    evaluated = []

    def scores():
        for index in range(48):
            evaluated.append(index)
            yield score

    assert vbench_cpu.motion_decision(scores(), pairs=48, threshold=0.5, required=12) == expected
    assert len(evaluated) == consumed


def test_appearance_uses_original_and_only_temporal_metrics_use_resized_copy(monkeypatch):
    calls = []

    def score(path, *, prompt, dimensions):
        calls.append((path, dimensions))
        return {metric: 0.99 for metric in dimensions}

    @contextmanager
    def reduced(path, width):
        assert path == "original.mp4" and width == 960
        yield "temporal.mp4"

    monkeypatch.setattr(vbench_cpu, "score_vbench", score)
    monkeypatch.setattr(vbench_cpu, "temporal_video", reduced)
    monkeypatch.setattr(vbench_cpu, "dynamic_degree", lambda path: 1.0 if path == "temporal.mp4" else -1)
    result = vbench_cpu.score_clip_ci("original.mp4", prompt="test")
    assert calls == [("original.mp4", vbench_cpu.APPEARANCE_METRICS), ("temporal.mp4", ["motion_smoothness"])]
    assert len(result) == 5 and result["dynamic_degree"] == 1.0


@pytest.fixture
def temporal_source(tmp_path):
    import av
    import numpy as np

    # Shell metacharacters are literal filename characters. No command may run.
    path = tmp_path / "-clip ; $(touch marker) `id`.mp4"
    expected = np.random.default_rng(0).integers(0, 256, size=(4, 24, 32, 3), dtype=np.uint8)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264rgb", rate=Fraction(24000, 1001))
        stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for pixels in expected:
            frame = av.VideoFrame.from_ndarray(pixels.repeat(2, axis=0).repeat(2, axis=1), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path, expected


def test_temporal_resize_is_lossless_and_runs_without_subprocesses(temporal_source, monkeypatch):
    import subprocess

    import av
    import numpy as np

    def forbid_process(*args, **kwargs):
        raise AssertionError("Temporal resize must not launch OS commands")

    monkeypatch.setattr(subprocess, "Popen", forbid_process)
    path, expected = temporal_source
    original = vbench_cpu.video_info(path)
    with av.open(str(path)) as source:
        timestamps = [frame.pts * frame.time_base for frame in source.decode(video=0)]
    with vbench_cpu.temporal_video(path, 32) as reduced:
        assert vbench_cpu.video_info(reduced) == (32, 24, *original[2:])
        with av.open(str(reduced)) as result:
            frames = list(result.decode(video=0))
        assert [frame.pts * frame.time_base for frame in frames] == timestamps
        assert np.array_equal(np.stack([frame.to_ndarray(format="rgb24") for frame in frames]), expected)
    assert not reduced.exists()
    assert path.exists()


@pytest.mark.parametrize("after", [(32, 24, 3, 24), (32, 24, 4, 12)])
def test_temporal_resize_cannot_drop_frames_or_change_fps(temporal_source, monkeypatch, after, expect_error):
    path, _ = temporal_source
    metadata = iter([(64, 48, 4, 24), after])
    monkeypatch.setattr(vbench_cpu, "video_info", lambda path: next(metadata))
    with expect_error(ValueError, "frame count or frame rate"):
        with vbench_cpu.temporal_video(path, 32):
            pass


@pytest.mark.parametrize("width", ["32; touch marker", 32.0, True, 0, -2, 31])
def test_temporal_width_rejects_non_integer_and_invalid_values(width, expect_error):
    with expect_error(ValueError, "positive even integer"):
        with vbench_cpu.temporal_video("unused.mp4", width):
            pass
