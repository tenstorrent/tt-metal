# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from loguru import logger


@dataclass(frozen=True)
class Audio:
    """Decoded audio: waveform + sampling rate. Produced by the LTX audio decode and
    consumed by ``export_video_audio``."""

    waveform: torch.Tensor
    sampling_rate: int


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


def export_video_audio(video_pixels: torch.Tensor, output_path: str, fps: int = 24, audio: Audio | None = None) -> None:
    """Export decoded video (and optionally audio) to MP4.

    Matches reference ltx_pipelines.utils.media_io.encode_video exactly:
    - H.264 video with yuv420p pixel format
    - AAC audio stream (if audio provided)
    - Correct [-1,1] -> uint8 conversion

    Args:
        video_pixels: (B, C, F, H, W) from decode_latents(), range [-1, 1]
        output_path: output .mp4 path
        fps: frame rate
        audio: decoded ``Audio`` (waveform + sampling rate), or None
    """
    import av

    # Convert to (F, H, W, C) uint8
    # In-place [-1,1] -> [0,255]: one fp32 copy + in-place passes instead of
    # allocating a fresh full-size tensor per arithmetic op (127.5 == 255/2).
    v = video_pixels[0].float()
    v.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
    # .contiguous() forces a single bulk (F,H,W,C) copy here, so each per-frame
    # slice below is already contiguous and VideoFrame.from_ndarray just wraps it
    # — otherwise the permuted view makes from_ndarray do a strided copy per frame.
    frames = v.to(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu().numpy()  # (F, H, W, C)

    _, height, width, _ = frames.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    # "veryfast" preset + multi-threaded encode is ~5-8x faster than libx264's
    # default "medium" single-threaded path, while crf 23 keeps the quality higher.
    stream.options = {"preset": "veryfast", "crf": "23"}
    stream.thread_type = "AUTO"

    # Prepare audio stream if provided
    audio_stream = None
    if audio is not None:
        audio_stream = container.add_stream("aac", rate=audio.sampling_rate)
        audio_stream.codec_context.sample_rate = audio.sampling_rate
        audio_stream.codec_context.layout = "stereo"
        audio_stream.codec_context.time_base = Fraction(1, audio.sampling_rate)

    # Write video frames
    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush video encoder
    for packet in stream.encode():
        container.mux(packet)

    # Write audio if provided
    if audio is not None and audio_stream is not None:
        samples = audio.waveform
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[1] != 2 and samples.shape[0] == 2:
            samples = samples.T
        if samples.shape[1] != 2:
            logger.warning(f"Audio has {samples.shape[1]} channels, expected 2 — duplicating mono")
            samples = samples[:, :1].repeat(1, 2)

        if samples.dtype != torch.int16:
            samples = torch.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).to(torch.int16)

        frame_in = av.AudioFrame.from_ndarray(
            samples.contiguous().reshape(1, -1).cpu().numpy(),
            format="s16",
            layout="stereo",
        )
        frame_in.sample_rate = audio.sampling_rate

        # Resample to encoder format and write
        cc = audio_stream.codec_context
        resampler = av.audio.resampler.AudioResampler(
            format=cc.format or "fltp",
            layout=cc.layout or "stereo",
            rate=cc.sample_rate or audio.sampling_rate,
        )
        for resampled in resampler.resample(frame_in):
            for packet in audio_stream.encode(resampled):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

    container.close()
    logger.info(f"Saved: {output_path} ({frames.shape[0]}f @ {fps}fps)")
