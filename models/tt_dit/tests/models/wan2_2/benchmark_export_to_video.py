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

import imageio_ffmpeg
import numpy as np
from diffusers.utils import export_to_video


def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.split("x")
    return int(w), int(h)


def run_iters(fn, iterations):
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        print(f"    iter {i}: {times[-1]:.3f}s")
    avg = sum(times) / len(times)
    print(f"    avg: {avg:.3f}s  min: {min(times):.3f}s  max: {max(times):.3f}s")
    return avg, min(times), max(times)


def bench_export_to_video(frames, iterations, fps):
    """Baseline: diffusers export_to_video (default quality=5, medium preset)."""
    print("  [baseline] export_to_video (libx264, quality=5, medium preset)")
    # warm-up
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        export_to_video(frames, f.name, fps=fps)

    def fn():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            export_to_video(frames, f.name, fps=fps)

    return run_iters(fn, iterations)


def bench_imageio_ultrafast(frames, iterations, fps):
    """imageio_ffmpeg directly with libx264 ultrafast preset — skips diffusers overhead,
    uses fastest x264 preset at the cost of compression ratio."""
    print("  [ultrafast] imageio_ffmpeg libx264, ultrafast preset, quality=5")
    width, height = frames[0].shape[1], frames[0].shape[0]

    def fn():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            gen = imageio_ffmpeg.write_frames(
                f.name,
                (width, height),
                fps=fps,
                quality=5,
                output_params=["-preset", "ultrafast"],
            )
            gen.send(None)
            for frame in frames:
                gen.send(frame)
            gen.close()

    # warm-up
    fn()
    return run_iters(fn, iterations)


def bench_imageio_superfast(frames, iterations, fps):
    """libx264 superfast preset — one step slower than ultrafast, better compression."""
    print("  [superfast] imageio_ffmpeg libx264, superfast preset, quality=5")
    width, height = frames[0].shape[1], frames[0].shape[0]

    def fn():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            gen = imageio_ffmpeg.write_frames(
                f.name,
                (width, height),
                fps=fps,
                quality=5,
                output_params=["-preset", "superfast"],
            )
            gen.send(None)
            for frame in frames:
                gen.send(frame)
            gen.close()

    fn()
    return run_iters(fn, iterations)


def bench_imageio_yuv444(frames, iterations, fps):
    """libx264 ultrafast with yuv444p — skips chroma subsampling, may be faster
    for random/noisy frames where subsampling adds little compression benefit."""
    print("  [yuv444]    imageio_ffmpeg libx264, ultrafast, yuv444p (no chroma subsampling)")
    width, height = frames[0].shape[1], frames[0].shape[0]

    def fn():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            gen = imageio_ffmpeg.write_frames(
                f.name,
                (width, height),
                fps=fps,
                quality=5,
                pix_fmt_out="yuv444p",
                output_params=["-preset", "ultrafast"],
            )
            gen.send(None)
            for frame in frames:
                gen.send(frame)
            gen.close()

    fn()
    return run_iters(fn, iterations)


def bench_imageio_high_crf(frames, iterations, fps):
    """libx264 ultrafast with high CRF (lower quality, less work per frame)."""
    print("  [high-crf]  imageio_ffmpeg libx264, ultrafast, CRF=35 (lower quality)")
    width, height = frames[0].shape[1], frames[0].shape[0]

    def fn():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            gen = imageio_ffmpeg.write_frames(
                f.name,
                (width, height),
                fps=fps,
                quality=None,
                output_params=["-preset", "ultrafast", "-crf", "35"],
            )
            gen.send(None)
            for frame in frames:
                gen.send(frame)
            gen.close()

    fn()
    return run_iters(fn, iterations)


def bench_videotoolbox(frames, iterations, fps):
    """Apple VideoToolbox hardware H.264 encoder (macOS / Apple Silicon only).
    Skipped gracefully on platforms where h264_videotoolbox is unavailable."""
    import subprocess

    print("  [videotoolbox] h264_videotoolbox (Apple hardware encoder, macOS only)")
    width, height = frames[0].shape[1], frames[0].shape[0]

    # Check availability
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
    )
    if "h264_videotoolbox" not in probe.stdout:
        print("    SKIPPED: h264_videotoolbox not available on this platform")
        return None

    def fn():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            gen = imageio_ffmpeg.write_frames(
                f.name,
                (width, height),
                fps=fps,
                codec="h264_videotoolbox",
                quality=None,
                output_params=["-b:v", "8M"],  # fixed bitrate; VT doesn't support CRF
            )
            gen.send(None)
            for frame in frames:
                gen.send(frame)
            gen.close()

    fn()
    return run_iters(fn, iterations)


def benchmark(num_frames: int, width: int, height: int, iterations: int, fps: int = 16):
    print(f"\n=== {width}x{height}, {num_frames} frames, fps={fps} ===")

    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8)

    results = {}
    results["baseline"] = bench_export_to_video(frames, iterations, fps)
    results["ultrafast"] = bench_imageio_ultrafast(frames, iterations, fps)
    results["superfast"] = bench_imageio_superfast(frames, iterations, fps)
    results["yuv444"] = bench_imageio_yuv444(frames, iterations, fps)
    results["high-crf"] = bench_imageio_high_crf(frames, iterations, fps)
    results["videotoolbox"] = bench_videotoolbox(frames, iterations, fps)

    print(f"\n  Summary ({width}x{height}):")
    base_avg = results["baseline"][0]
    for name, result in results.items():
        if result is None:
            print(f"    {name:<14} SKIPPED")
            continue
        avg, mn, mx = result
        speedup = base_avg / avg
        print(f"    {name:<14} avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s  speedup={speedup:.2f}x")


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
