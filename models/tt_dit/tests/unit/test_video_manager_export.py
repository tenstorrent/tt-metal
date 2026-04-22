# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import statistics
import time

import numpy as np
import pytest

from models.tt_dit.utils.video_manager import VideoManager

X264_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]

NUM_FRAMES = 81
HEIGHT = 720
WIDTH = 1280
NUM_RUNS = 10


@pytest.fixture(scope="module")
def random_frames():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def video_manager():
    return VideoManager()


@pytest.mark.parametrize("preset", X264_PRESETS)
def test_export_to_mp4_preset(preset, random_frames, video_manager, monkeypatch, tmp_path):
    """Benchmark VideoManager.export_to_mp4 across x264 presets (10 runs each)."""
    monkeypatch.setenv("TT_VIDEO_EXPORT_PRESET", preset)
    monkeypatch.setenv("TT_VIDEO_EXPORT_CRF", "25")
    monkeypatch.setattr("models.tt_dit.utils.video_manager._VIDEO_OUTPUT_DIR", tmp_path)

    durations = []
    last_path = None

    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        last_path = video_manager.export_to_mp4(random_frames, fps=16)
        durations.append(time.perf_counter() - t0)

    file_size_mb = os.path.getsize(last_path) / (1024 * 1024)

    mean_t = statistics.mean(durations)
    std_t = statistics.stdev(durations)
    min_t = min(durations)
    max_t = max(durations)
    median_t = statistics.median(durations)

    print(
        f"\n{'=' * 80}\n"
        f"Preset: {preset:12s}\n"
        f"  Runs       : {NUM_RUNS}\n"
        f"  Mean       : {mean_t:8.3f}s\n"
        f"  Median     : {median_t:8.3f}s\n"
        f"  Std        : {std_t:8.3f}s\n"
        f"  Min        : {min_t:8.3f}s\n"
        f"  Max        : {max_t:8.3f}s\n"
        f"  File size  : {file_size_mb:8.2f} MB\n"
        f"{'=' * 80}"
    )


def test_export_to_mp4_summary(random_frames, video_manager, monkeypatch, tmp_path):
    """Run all presets and print a combined summary table."""
    monkeypatch.setenv("TT_VIDEO_EXPORT_CRF", "25")
    monkeypatch.setattr("models.tt_dit.utils.video_manager._VIDEO_OUTPUT_DIR", tmp_path)

    rows = []

    for preset in X264_PRESETS:
        monkeypatch.setenv("TT_VIDEO_EXPORT_PRESET", preset)
        durations = []
        last_path = None

        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            last_path = video_manager.export_to_mp4(random_frames, fps=16)
            durations.append(time.perf_counter() - t0)

        file_size_mb = os.path.getsize(last_path) / (1024 * 1024)

        rows.append(
            {
                "preset": preset,
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "std": statistics.stdev(durations),
                "min": min(durations),
                "max": max(durations),
                "size_mb": file_size_mb,
            }
        )

    header = f"{'Preset':>12s} | {'Mean':>8s} | {'Median':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s} | {'Size MB':>8s}"
    sep = "-" * len(header)
    lines = ["\n" + sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['preset']:>12s} | {r['mean']:8.3f} | {r['median']:8.3f} | {r['std']:8.3f} | {r['min']:8.3f} | {r['max']:8.3f} | {r['size_mb']:8.2f}"
        )
    lines.append(sep)

    print("\n".join(lines))
