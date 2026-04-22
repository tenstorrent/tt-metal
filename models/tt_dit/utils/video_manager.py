# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import functools
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_MIN_CRF = 0
_MAX_CRF = 51
_FFMPEG_ENCODE_TIMEOUT_S = 600
_FFMPEG_REMUX_TIMEOUT_S = 60
_VIDEO_OUTPUT_DIR = Path("/tmp/videos")
_VALID_CHANNEL_COUNTS = (1, 3, 4)
_RGB_CHANNELS = 3
_MAX_PIXEL_VALUE = 255.0
_NORMALIZED_RANGE_MAX = 1.0

_logger = logging.getLogger(__name__)


def _log_execution_time(label: str):
    """Lightweight replacement for utils.decorators.log_execution_time."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            _logger.debug("%s took %.3fs", label, time.perf_counter() - t0)
            return result

        return wrapper

    return decorator


class VideoManager:
    """MP4 export via FFmpeg subprocess pipe (raw RGB -> libx264)."""

    @_log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames: NDArray, fps: int = 16) -> str:
        """
        Export frames to MP4 (H.264 via ffmpeg).

        Env (optional):
            TT_VIDEO_EXPORT_CRF: 0-51, lower = better quality. Default 23.
            TT_VIDEO_EXPORT_PRESET: ultrafast ... veryslow. Default medium.
        """
        if hasattr(frames, "frames"):
            frames = frames.frames

        _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")

        crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "23"))
        crf = max(_MIN_CRF, min(_MAX_CRF, crf))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "ultrafast").strip()

        try:
            processed = self._process_frames_for_export(frames)
            cmd = self._build_encode_cmd(processed, output_path, fps, crf, preset)
            self._run_ffmpeg(cmd, stdin_data=processed.tobytes())
            return output_path

        except Exception as e:
            _logger.error("Video export failed: %s", e)
            raise RuntimeError(f"Failed to export video: {e}") from e

    @_log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames: NDArray) -> NDArray[np.uint8]:
        """Normalize to contiguous uint8 (N, H, W, 3) for rawvideo rgb24."""
        frames = _normalize_shape(frames)
        frames = _normalize_channels(frames)
        frames = _normalize_dtype(frames)

        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)

        return frames

    @staticmethod
    def _build_encode_cmd(
        frames: NDArray, output_path: str, fps: int, crf: int, preset: str
    ) -> list[str]:
        """Build the ffmpeg rawvideo -> libx264 command list."""
        _, height, width, channels = frames.shape
        if channels != _RGB_CHANNELS:
            raise ValueError(
                f"Expected {_RGB_CHANNELS} RGB channels after processing, got {channels}"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
        ]

        if crf == 0:
            cmd.extend(["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-tune",
                    "film",
                    "-profile:v",
                    "high",
                    "-level",
                    "4.2",
                ]
            )

        if preset:
            cmd.extend(["-preset", preset])

        cmd.extend(["-movflags", "+faststart", output_path])
        return cmd

    @staticmethod
    def _run_ffmpeg(
        cmd: list[str],
        stdin_data: bytes | None = None,
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Execute an ffmpeg command, raising on failure or timeout."""
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            _, stderr = process.communicate(input=stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg export timed out") from None

        if process.returncode != 0:
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @classmethod
    def ensure_faststart(cls, input_path: str, output_path: str) -> None:
        """Rewrites the MP4 file with -movflags faststart using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c",
            "copy",
            "-movflags",
            "faststart",
            output_path,
        ]
        cls._run_ffmpeg(cmd, timeout=_FFMPEG_REMUX_TIMEOUT_S)


def _normalize_shape(frames: NDArray) -> NDArray:
    """Squeeze batch dim and validate 4D (N, H, W, C)."""
    if frames.ndim == 5:
        frames = frames[0]

    if frames.ndim != 4:
        raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

    return frames


def _normalize_channels(frames: NDArray) -> NDArray:
    """Convert grayscale or RGBA to RGB."""
    _, _, _, channels = frames.shape

    if channels not in _VALID_CHANNEL_COUNTS:
        raise ValueError(f"Frames have {channels} channels, expected 1, 3, or 4")

    if channels == 1:
        return np.repeat(frames, _RGB_CHANNELS, axis=-1)
    if channels == 4:
        return frames[..., :_RGB_CHANNELS]

    return frames


def _normalize_dtype(frames: NDArray) -> NDArray[np.uint8]:
    """Convert to uint8, handling float [0,1] and [0,255] ranges."""
    if frames.dtype == np.uint8:
        return frames

    if frames.dtype in (np.float32, np.float64):
        max_val = float(np.max(frames)) if frames.size else 0.0
        if max_val <= _NORMALIZED_RANGE_MAX:
            return (frames * _MAX_PIXEL_VALUE).clip(0, 255).astype(np.uint8)
        return frames.clip(0, 255).astype(np.uint8)

    return frames.clip(0, 255).astype(np.uint8)
