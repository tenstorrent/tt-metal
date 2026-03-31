# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Benchmark diffusers export_to_video MP4 encoding time.

Generates synthetic frames and measures how long export_to_video takes
to encode them to MP4. No TT device or model weights required.

Usage:
    python benchmark_export_to_video.py
    python benchmark_export_to_video.py --num_frames 81 --iterations 5 --resolutions 832x480 1280x720
"""

import argparse
import tempfile
import time

import numpy as np
from diffusers.utils import export_to_video


def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.split("x")
    return int(w), int(h)


def benchmark(num_frames: int, width: int, height: int, iterations: int, fps: int = 16):
    print(f"\n--- {width}x{height}, {num_frames} frames, fps={fps} ---")

    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8)

    # Warm-up
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        export_to_video(frames, f.name, fps=fps)
    print("Warm-up done")

    times = []
    for i in range(iterations):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            t0 = time.perf_counter()
            export_to_video(frames, f.name, fps=fps)
            elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  iter {i}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    print(f"  avg: {avg:.3f}s  min: {min(times):.3f}s  max: {max(times):.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark export_to_video encoding time")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=str,
        default=["832x480", "1280x720"],
        help="Resolutions as WxH (e.g. 832x480 1280x720)",
    )
    args = parser.parse_args()

    for res in args.resolutions:
        width, height = parse_resolution(res)
        benchmark(args.num_frames, width, height, args.iterations, args.fps)


if __name__ == "__main__":
    main()
