# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Frame/GIF/MP4 writeout for generated HunyuanVideo-1.5 output.

Writeout is pure host cost and it is not small at production length. Measured on
the reference 64-thread EPYC host with real 480x848 generated frames:

    frames   serial PNG   16-thread PNG   Pillow GIF   mp4
    13       2.34 s       0.21 s          1.70 s       0.22 s
    121      20.59 s      1.50 s          15.67 s      0.82 s

PNG encoding is embarrassingly parallel and Pillow releases the GIL inside the
zlib encoder, so threads give a near-linear win with byte-identical files. The
animated GIF is a serial Pillow loop and, at 121 frames, is a 26.8 MB artifact
that carries no information the mp4 does not.

Everything here defaults to today's behaviour. ``HY_FAST_WRITEOUT=1`` opts into
threaded PNG encoding; ``HY_PNG_COMPRESS`` overrides the zlib level; and
``HY_SAVE_GIF=0`` drops the GIF.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

PILLOW_DEFAULT_PNG_COMPRESS = 6


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


def _as_pil(frames):
    import numpy as np
    from PIL import Image

    return [
        frame
        if isinstance(frame, Image.Image)
        else Image.fromarray((np.asarray(frame).clip(0, 1) * 255).astype("uint8"))
        for frame in frames
    ]


def _png_workers(count: int) -> int:
    return max(1, min(count, os.cpu_count() or 1))


def write_png_frames(frames, outdir, *, compress_level=None, threaded=False) -> list[str]:
    """Write ``frame_%03d.png``. Threaded and serial output are byte-identical.

    zlib is deterministic for a fixed level, and each frame is an independent
    file, so parallelism changes only wall time. ``compress_level=None`` omits
    the argument entirely, which is exactly what the previous inline writeout
    did, so the unconfigured path cannot drift from it.
    """
    paths = [os.path.join(outdir, f"frame_{index:03d}.png") for index in range(len(frames))]
    options = {} if compress_level is None else {"compress_level": compress_level}

    def save(pair):
        index, frame = pair
        frame.save(paths[index], **options)

    if threaded and len(frames) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=_png_workers(len(frames))) as pool:
            list(pool.map(save, enumerate(frames)))
    else:
        for pair in enumerate(frames):
            save(pair)
    return paths


def save_generated_frames(
    frames,
    outdir,
    *,
    fps: int = 24,
    fast: bool | None = None,
    compress_level: int | None = None,
    write_gif: bool | None = None,
    write_mp4: bool = True,
) -> dict:
    """Persist one generation's output and report per-artifact seconds.

    The returned timings are what a caller should log; the files produced are
    identical to the previous inline implementation unless a gate is flipped.
    """
    if fast is None:
        fast = _env_flag("HY_FAST_WRITEOUT", "0")
    if compress_level is None and "HY_PNG_COMPRESS" in os.environ:
        compress_level = int(os.environ["HY_PNG_COMPRESS"])
    if write_gif is None:
        write_gif = _env_flag("HY_SAVE_GIF", "1")
    if compress_level is not None and not 0 <= compress_level <= 9:
        raise ValueError(f"HY_PNG_COMPRESS must be 0..9, got {compress_level}")

    os.makedirs(outdir, exist_ok=True)
    pil = _as_pil(frames)
    timings = {
        "frames": len(pil),
        "fast": bool(fast),
        "png_compress_level": PILLOW_DEFAULT_PNG_COMPRESS if compress_level is None else compress_level,
    }

    start = time.perf_counter()
    write_png_frames(pil, outdir, compress_level=compress_level, threaded=fast)
    timings["png_s"] = round(time.perf_counter() - start, 3)

    timings["gif"] = bool(write_gif)
    if write_gif:
        start = time.perf_counter()
        pil[0].save(
            os.path.join(outdir, "tt_blackhole.gif"),
            save_all=True,
            append_images=pil[1:],
            duration=125,
            loop=0,
        )
        timings["gif_s"] = round(time.perf_counter() - start, 3)

    ffmpeg = shutil.which("ffmpeg")
    timings["mp4"] = bool(write_mp4 and ffmpeg)
    if write_mp4 and ffmpeg:
        start = time.perf_counter()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(outdir, "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                os.path.join(outdir, "tt_blackhole.mp4"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        timings["mp4_s"] = round(time.perf_counter() - start, 3)

    timings["total_s"] = round(sum(v for k, v in timings.items() if k.endswith("_s")), 3)
    return timings
