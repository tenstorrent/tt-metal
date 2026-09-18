# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
from contextlib import contextmanager
from types import SimpleNamespace

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


@pytest.mark.parametrize("after", [(960, 544, 144, 24), (960, 544, 145, 12)])
def test_temporal_resize_cannot_drop_frames_or_change_fps(monkeypatch, after, expect_error):
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"))
    metadata = iter([(1920, 1088, 145, 24), after])
    monkeypatch.setattr(vbench_cpu, "video_info", lambda path: next(metadata))
    monkeypatch.setattr(vbench_cpu.subprocess, "run", lambda *args, **kwargs: None)
    with expect_error(ValueError, "frame count or frame rate"):
        with vbench_cpu.temporal_video("original.mp4", 960):
            pass
