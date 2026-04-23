# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess

import numpy as np


def rgb_to_yuv420p(frames_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB frames (T, H, W, 3) to planar YUV420p byte layout.

    Output shape is (T, H*W*3//2), uint8, with each row containing Y plane
    (H*W bytes) followed by U plane (H*W/4) then V plane (H*W/4). BT.601
    limited-range coefficients are used.

    Chroma 2×2 subsampling is done in **float**, before the clip+cast to
    uint8, matching the on-device path. This preserves more precision than
    the alternative (clip+cast first, then average uint8), where each pixel
    pays two rounding steps and the clip can saturate mid-pipeline.
    """
    frames_rgb = np.asarray(frames_rgb)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames (T, H, W, 3), got {frames_rgb.shape}")
    t, h, w, _ = frames_rgb.shape
    if h % 2 or w % 2:
        raise ValueError(f"yuv420p requires even H and W, got {h}x{w}")

    if frames_rgb.dtype != np.uint8:
        frames_rgb = (frames_rgb * 255).clip(0, 255).astype(np.uint8)

    r = frames_rgb[..., 0].astype(np.float32)
    g = frames_rgb[..., 1].astype(np.float32)
    b = frames_rgb[..., 2].astype(np.float32)

    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128.0
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128.0

    # Subsample in float, then clip+cast at the very end.
    u_ds = u.reshape(t, h // 2, 2, w // 2, 2).mean(axis=(2, 4))
    v_ds = v.reshape(t, h // 2, 2, w // 2, 2).mean(axis=(2, 4))

    y = y.clip(0, 255).astype(np.uint8)
    u_ds = u_ds.clip(0, 255).astype(np.uint8)
    v_ds = v_ds.clip(0, 255).astype(np.uint8)

    y_size = h * w
    uv_size = (h * w) // 4
    out = np.empty((t, y_size + 2 * uv_size), dtype=np.uint8)
    out[:, :y_size] = y.reshape(t, -1)
    out[:, y_size : y_size + uv_size] = u_ds.reshape(t, -1)
    out[:, y_size + uv_size :] = v_ds.reshape(t, -1)
    return out


def export_to_video(
    frames,
    output_video_path: str,
    fps: int = 16,
    crf: int = 25,
    preset: str | None = "ultrafast",
    pix_fmt_in: str = "rgb24",
    width: int | None = None,
    height: int | None = None,
    gop: int | None = None,
    threads: int | None = None,
    tune: str | None = None,
) -> str:
    """Encode frames to video via ffmpeg subprocess.

    For ``pix_fmt_in="rgb24"`` (default), ``frames`` must have shape
    (T, H, W, 3) as uint8 in [0,255] or float in [0,1]. For
    ``pix_fmt_in="yuv420p"``, ``frames`` must be uint8 with each frame's
    planar YUV bytes laid out contiguously (e.g. the output of
    :func:`rgb_to_yuv420p`), and ``width``/``height`` must be provided.

    ``preset`` is passed as libx264 ``-preset``. ``gop`` maps to ``-g``
    (keyframe interval). ``threads`` maps to ``-threads``.
    """
    from imageio_ffmpeg import get_ffmpeg_exe

    frames = np.asarray(frames)

    if pix_fmt_in == "rgb24":
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames (T, H, W, 3), got {frames.shape}")
        _, h, w, _ = frames.shape
    else:
        if width is None or height is None:
            raise ValueError(f"width and height required for pix_fmt_in={pix_fmt_in!r}")
        if frames.dtype != np.uint8:
            raise ValueError(f"pix_fmt_in={pix_fmt_in!r} requires uint8 frames, got {frames.dtype}")
        h, w = height, width

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
        pix_fmt_in,
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
    if tune is not None:
        cmd += ["-tune", tune]
    if gop is not None:
        cmd += ["-g", str(gop)]
    if threads is not None:
        cmd += ["-threads", str(threads)]
    cmd += ["-v", "warning", output_video_path]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Grow the kernel pipe buffer so a full frame fits without blocking the
    # writer between partial kernel writes. Default on Linux is 64 KiB; the
    # unprivileged cap (/proc/sys/fs/pipe-max-size) is typically 1 MiB. Silent
    # fallback on non-Linux or if the cap rejects the request.
    _F_SETPIPE_SZ = 1031
    try:
        import fcntl

        fcntl.fcntl(p.stdin.fileno(), _F_SETPIPE_SZ, 1 << 20)
    except (OSError, ValueError, ImportError, AttributeError):
        pass

    try:
        if pix_fmt_in == "rgb24" and frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        # Bypass io.BufferedWriter and write straight to the raw fd. At frame
        # sizes >1 MB BufferedWriter passes through anyway, but skipping its
        # bookkeeping removes one Python-side layer from the per-frame path.
        fd = p.stdin.fileno()
        for frame in frames:
            mv = memoryview(np.ascontiguousarray(frame)).cast("B")
            pos, n = 0, len(mv)
            while pos < n:
                pos += os.write(fd, mv[pos:])
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


def export_to_video_pyav(
    frames,
    output_video_path: str,
    fps: int = 16,
    crf: int = 25,
    preset: str = "ultrafast",
    pix_fmt_in: str = "rgb24",
    width: int | None = None,
    height: int | None = None,
    gop: int | None = None,
    threads: int | None = None,
    tune: str | None = None,
) -> str:
    """In-process encode via PyAV (libavcodec bindings). No subprocess, no pipe.

    Frame data is wrapped as av.VideoFrame and passed straight to libx264 in
    the same process — eliminates the fork/exec, kernel pipe, and all
    user->kernel write copies. ``pix_fmt_in`` selects whether frames arrive as
    rgb24 (shape (T,H,W,3), libswscale converts internally) or yuv420p (shape
    (T, H*W*3//2) planar bytes; reshaped to (H*3/2, W) per frame for PyAV).
    """
    import av

    frames = np.asarray(frames)

    if pix_fmt_in == "rgb24":
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames (T, H, W, 3), got {frames.shape}")
        _, h, w, _ = frames.shape
    else:
        if width is None or height is None:
            raise ValueError(f"width and height required for pix_fmt_in={pix_fmt_in!r}")
        if frames.dtype != np.uint8:
            raise ValueError(f"pix_fmt_in={pix_fmt_in!r} requires uint8 frames, got {frames.dtype}")
        h, w = height, width

    container = av.open(output_video_path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        opts = {"crf": str(crf), "preset": preset}
        if tune is not None:
            opts["tune"] = tune
        stream.options = opts
        if gop is not None:
            stream.gop_size = gop
        if threads is not None:
            # thread_count alone leaves thread_type=NONE (single-threaded).
            # FRAME threading is what gives libx264 its parallelism at fast
            # presets; AUTO lets libav pick but often lands on SLICE, which
            # scales worse at high thread counts.
            stream.codec_context.thread_type = "FRAME"
            stream.codec_context.thread_count = threads

        if pix_fmt_in == "rgb24" and frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)

        for i in range(frames.shape[0]):
            if pix_fmt_in == "rgb24":
                vframe = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
            else:
                # Reshape flat (H*W*3/2,) planar buffer to (H*3/2, W) which is
                # PyAV's expected yuv420p layout. This is a view, no copy.
                vframe = av.VideoFrame.from_ndarray(frames[i].reshape(h * 3 // 2, w), format="yuv420p")
            for packet in stream.encode(vframe):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    return output_video_path
