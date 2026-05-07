# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import numpy as np


def export_to_video(
    frames,
    output_video_path: str,
    fps: int = 16,
    crf: int = 25,
    preset: str | None = "ultrafast",
    *,
    pix_fmt: str = "rgb24",
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Encode frames to video via ffmpeg subprocess.

    Two supported input layouts:

      * ``pix_fmt="rgb24"`` (default): ``frames`` has shape ``(T, H, W, 3)``,
        either ``uint8`` (passed straight through) or float in ``[0, 1]``
        (scaled to uint8).
      * ``pix_fmt="yuv420p"``: ``frames`` has shape
        ``(T, H*W + 2*(H/2 * W/2))`` ``uint8``, which is the flattened YUV
        4:2:0 planar layout that
        :func:`models.tt_dit.utils.tensor.fast_device_to_host_yuv` produces.
        The byte buffer alone doesn't carry image dimensions, so ``width`` and
        ``height`` are required in this mode.

    ``preset`` is passed directly as the libx264 ``-preset`` flag (e.g.
    "ultrafast", "veryfast", "medium", "slow").  When *None* ffmpeg uses its
    built-in default ("medium"). We default to "ultrafast" for faster encoding,
    at expense of filesize.
    """
    from imageio_ffmpeg import get_ffmpeg_exe

    frames = np.asarray(frames)

    if pix_fmt == "yuv420p":
        if width is None or height is None:
            raise ValueError("export_to_video(pix_fmt='yuv420p') requires explicit `width` and `height`")
        if frames.ndim != 2 or frames.dtype != np.uint8:
            raise ValueError(
                f"yuv420p frames must be (T, planar_bytes) uint8, " f"got shape {frames.shape} dtype {frames.dtype}"
            )
        expected_planar = height * width + 2 * (height // 2) * (width // 2)
        if frames.shape[1] != expected_planar:
            raise ValueError(
                f"yuv420p frames second dim must be {expected_planar} for " f"{width}x{height}, got {frames.shape[1]}"
            )
        h, w = height, width
    elif pix_fmt == "rgb24":
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames with shape (T, H, W, 3), got {frames.shape}")
        _, h, w, _ = frames.shape
    else:
        raise ValueError(f"Unsupported pix_fmt: {pix_fmt!r}")

    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        pix_fmt,
        "-r",
        f"{fps:.2f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
    ]
    if preset is not None:
        cmd += ["-preset", preset]
    cmd += [
        "-v",
        "warning",
        output_video_path,
    ]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for frame in frames:
            if pix_fmt == "rgb24" and frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            # For yuv420p, ``frame`` is already the per-frame [Y|Cb|Cr] uint8
            # byte buffer — write it through as-is.
            p.stdin.write(frame.tobytes())
        p.stdin.close()
        stderr = p.stderr.read().decode("utf-8", errors="ignore")
        rc = p.wait()
    except Exception:
        p.kill()
        p.wait()
        raise

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with return code {rc}:\n{stderr}")

    return output_video_path
