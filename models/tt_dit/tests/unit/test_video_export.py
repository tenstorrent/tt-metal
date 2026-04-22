# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import statistics
import time

import numpy as np
import pytest

from models.tt_dit.utils.video import export_to_video, export_to_video_pyav, rgb_to_yuv420p

NUM_FRAMES = 81
HEIGHT = 720
WIDTH = 1280
FPS = 16
PRESET = "ultrafast"

GOPS = [1, 2, 4, 8, 16, 32]
THREADS = [1, 2, 4, 8, 10, 12, 14, 16, 18, 20]


@pytest.fixture(scope="module")
def frames_rgb():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def frames_yuv420(frames_rgb):
    return rgb_to_yuv420p(frames_rgb)


def _benchmark(frames, tmp_path, prefix, num_runs, encoder=export_to_video, **kwargs):
    durations = []
    out_path = None
    for i in range(num_runs):
        out_path = str(tmp_path / f"{prefix}_{i}.mp4")
        t0 = time.perf_counter()
        encoder(frames, out_path, fps=FPS, preset=PRESET, **kwargs)
        durations.append(time.perf_counter() - t0)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    return {
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "std": statistics.stdev(durations) if num_runs > 1 else 0.0,
        "min": min(durations),
        "max": max(durations),
        "size_mb": size_mb,
    }


# ---------------------------------------------------------------------------
# Sweep 1: RGB vs YUV420 input (skip ffmpeg's internal colorspace conversion).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pix_fmt_in", ["rgb24", "yuv420p"])
def test_pix_fmt_in(pix_fmt_in, frames_rgb, frames_yuv420, tmp_path):
    frames = frames_rgb if pix_fmt_in == "rgb24" else frames_yuv420
    stats = _benchmark(
        frames,
        tmp_path,
        f"pix_{pix_fmt_in}",
        num_runs=10,
        pix_fmt_in=pix_fmt_in,
        width=WIDTH,
        height=HEIGHT,
    )
    print(
        f"\n[pix_fmt_in={pix_fmt_in}] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


# ---------------------------------------------------------------------------
# Sweep 2+3: GOP x threads (combined — libx264 frame-threading parallelizes
# across GOPs so the two parameters interact). Single test, single table.
# ---------------------------------------------------------------------------


def test_gop_threads_sweep(frames_yuv420, tmp_path):
    grid = {}  # (gop, threads) -> stats
    num_runs = 10

    for gop in GOPS:
        for threads in THREADS:
            grid[(gop, threads)] = _benchmark(
                frames_yuv420,
                tmp_path,
                f"g{gop}_t{threads}",
                num_runs=num_runs,
                pix_fmt_in="yuv420p",
                width=WIDTH,
                height=HEIGHT,
                gop=gop,
                threads=threads,
            )

    lines = [
        "",
        f"gop x threads sweep (preset={PRESET}, yuv420p input, {NUM_FRAMES} frames, "
        f"{WIDTH}x{HEIGHT}, {num_runs} runs/cell)",
        "",
        "mean encode time (seconds)",
    ]
    header = "  gop\\threads | " + " | ".join(f"{t:>7d}" for t in THREADS)
    lines.append(header)
    lines.append("-" * len(header))
    for gop in GOPS:
        row = f"  {gop:>11d} | " + " | ".join(f"{grid[(gop, t)]['mean']:7.3f}" for t in THREADS)
        lines.append(row)

    lines.append("")
    lines.append("stdev encode time (seconds)")
    lines.append(header)
    lines.append("-" * len(header))
    for gop in GOPS:
        row = f"  {gop:>11d} | " + " | ".join(f"{grid[(gop, t)]['std']:7.3f}" for t in THREADS)
        lines.append(row)

    lines.append("")
    lines.append("output file size (MB)")
    lines.append(header)
    lines.append("-" * len(header))
    for gop in GOPS:
        row = f"  {gop:>11d} | " + " | ".join(f"{grid[(gop, t)]['size_mb']:7.2f}" for t in THREADS)
        lines.append(row)

    best_gop, best_th = min(grid, key=lambda k: grid[k]["mean"])
    best = grid[(best_gop, best_th)]
    lines.append("")
    lines.append(
        f"Fastest: gop={best_gop} threads={best_th} " f"mean={best['mean']:.3f}s size={best['size_mb']:.2f} MB"
    )

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# zerolatency tune: disables lookahead and B-frames in libx264. Expected to
# speed up encode at the cost of compression efficiency.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Best config: yuv420p input, gop=4, threads=16. Baseline for future
# comparisons against feeding-path or encoder changes.
# ---------------------------------------------------------------------------


def test_best_config(frames_yuv420, tmp_path):
    stats = _benchmark(
        frames_yuv420,
        tmp_path,
        "best",
        num_runs=10,
        pix_fmt_in="yuv420p",
        width=WIDTH,
        height=HEIGHT,
        gop=4,
        threads=16,
    )
    print(
        f"\n[best yuv420p gop=4 threads=16] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


# ---------------------------------------------------------------------------
# PyAV: in-process libavcodec. No subprocess, no pipe, no user->kernel write
# copies. Same config as test_best_config for apples-to-apples comparison.
# ---------------------------------------------------------------------------


def test_best_config_pyav(frames_yuv420, tmp_path):
    stats = _benchmark(
        frames_yuv420,
        tmp_path,
        "best_pyav",
        num_runs=10,
        encoder=export_to_video_pyav,
        pix_fmt_in="yuv420p",
        width=WIDTH,
        height=HEIGHT,
        gop=4,
        threads=16,
    )
    print(
        f"\n[pyav yuv420p gop=4 threads=16] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


@pytest.mark.parametrize("tune", [None, "zerolatency"])
def test_tune_zerolatency(tune, frames_yuv420, tmp_path):
    label = "off" if tune is None else tune
    stats = _benchmark(
        frames_yuv420,
        tmp_path,
        f"tune_{label}",
        num_runs=10,
        pix_fmt_in="yuv420p",
        width=WIDTH,
        height=HEIGHT,
        threads=16,
        tune=tune,
    )
    print(
        f"\n[tune={label}] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )
