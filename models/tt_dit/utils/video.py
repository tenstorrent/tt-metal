# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import shutil
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


def export_to_video_with_audio(frames, output_path: str, *, audio_path: str | None = None, fps: int = 16) -> str:
    """Write video frames and optionally mux an audio track into the same file."""
    export_to_video(frames, output_path, fps=fps)
    if audio_path is not None:
        merge_video_audio(output_path, audio_path)
    return output_path


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # set logging
    logging.basicConfig(level=logging.INFO)

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    from imageio_ffmpeg import get_ffmpeg_exe

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            get_ffmpeg_exe(),
            "-y",  # overwrite
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",  # copy video stream
            "-c:a",
            "aac",  # use AAC audio encoder
            "-b:a",
            "192k",  # set audio bitrate (optional)
            "-map",
            "0:v:0",  # select the first video stream
            "-map",
            "1:a:0",  # select the first audio stream
            "-shortest",  # choose the shortest duration
            temp_output,
        ]

        # execute the command
        logging.info("Start merging video and audio...")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logging.info(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        logging.error(f"merge_video_audio failed with error: {e}")
