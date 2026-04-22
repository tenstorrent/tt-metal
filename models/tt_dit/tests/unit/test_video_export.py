# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import statistics
import tempfile
import time

import numpy as np
import pytest

from models.tt_dit.utils.video import export_to_video

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


@pytest.mark.parametrize("preset", X264_PRESETS)
def test_export_to_video_preset(preset, random_frames, tmp_path):
    """Benchmark export_to_video across x264 presets (10 runs each)."""
    durations = []

    for i in range(NUM_RUNS):
        out_path = str(tmp_path / f"{preset}_{i}.mp4")
        t0 = time.perf_counter()
        export_to_video(random_frames, out_path, fps=16, preset=preset)
        durations.append(time.perf_counter() - t0)

    last_file = str(tmp_path / f"{preset}_{NUM_RUNS - 1}.mp4")
    file_size_mb = os.path.getsize(last_file) / (1024 * 1024)

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


def test_export_to_video_summary(random_frames, tmp_path):
    """Run all presets and print a combined summary table."""
    rows = []

    for preset in X264_PRESETS:
        durations = []
        for i in range(NUM_RUNS):
            out_path = str(tmp_path / f"summary_{preset}_{i}.mp4")
            t0 = time.perf_counter()
            export_to_video(random_frames, out_path, fps=16, preset=preset)
            durations.append(time.perf_counter() - t0)

        last_file = str(tmp_path / f"summary_{preset}_{NUM_RUNS - 1}.mp4")
        file_size_mb = os.path.getsize(last_file) / (1024 * 1024)

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
