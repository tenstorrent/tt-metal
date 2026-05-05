# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import numpy as np


def export_to_video(
    frames, output_video_path: str, fps: int = 16, crf: int = 25, preset: str | None = "ultrafast"
) -> str:
    """Encode frames to video via ffmpeg subprocess.

    Accepts either float32 [0,1] or uint8 [0,255] frames with shape (T, H, W, 3).
    ``preset`` is passed directly as the libx264 ``-preset`` flag (e.g.
    "ultrafast", "veryfast", "medium", "slow").  When *None* ffmpeg uses its
    built-in default ("medium"). We default to "ultrafast" for faster encoding,
    at expense of filesize.
    """
    from imageio_ffmpeg import get_ffmpeg_exe

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T, H, W, 3), got {frames.shape}")

    t, h, w, c = frames.shape

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
        "rgb24",
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
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
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
