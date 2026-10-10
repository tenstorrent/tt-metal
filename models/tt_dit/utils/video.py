# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import threading
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


def _add_audio_stream(container, audio: Audio | None):
    """Add an AAC audio stream to an open write container. Must run before any packet is muxed
    (PyAV forbids adding streams once muxing starts). Returns the stream, or None if no audio."""
    if audio is None:
        return None
    audio_stream = container.add_stream("aac", rate=audio.sampling_rate)
    audio_stream.codec_context.sample_rate = audio.sampling_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio.sampling_rate)
    return audio_stream


def _mux_audio(container, audio_stream, audio: Audio) -> None:
    """Encode + mux the decoded waveform into ``audio_stream`` (created by ``_add_audio_stream``)."""
    for packet in _encode_audio(audio_stream, audio):
        container.mux(packet)


def _encode_audio(audio_stream, audio: Audio) -> list:
    """Encode the decoded waveform into ``audio_stream``'s AAC packets without muxing them."""
    import av

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

    cc = audio_stream.codec_context
    resampler = av.audio.resampler.AudioResampler(
        format=cc.format or "fltp",
        layout=cc.layout or "stereo",
        rate=cc.sample_rate or audio.sampling_rate,
    )
    packets = []
    for resampled in resampler.resample(frame_in):
        packets.extend(audio_stream.encode(resampled))
    packets.extend(audio_stream.encode())
    return packets


def _x264_options() -> dict[str, str]:
    """libx264 options for the exports. ``LTX_EXPORT_LOSSLESS=1`` (parity/testing only) encodes losslessly so the
    decoded frames equal the pre-encode frames bit for bit and a comparison measures the pipeline, not the codec.

    The export is on the request's critical path. On a 1080p 145-frame clip, ultrafast at crf 20 encodes in about
    0.15 s against 0.65 s for veryfast at crf 23, and lands closer to the source frames (Y PSNR 47.9 dB vs 45.6 dB);
    the cost is a ~3.5x larger file. ``LTX_EXPORT_PRESET=veryfast`` restores the old veryfast/crf 23 export;
    ``LTX_EXPORT_CRF`` overrides the crf of either preset."""
    if os.environ.get("LTX_EXPORT_LOSSLESS", "0") != "0":
        return {"preset": "veryfast", "qp": "0"}
    preset = os.environ.get("LTX_EXPORT_PRESET", "ultrafast")
    return {"preset": preset, "crf": os.environ.get("LTX_EXPORT_CRF", "23" if preset == "veryfast" else "20")}


def _dump_audio_sidecar(output_path: str, audio: "Audio | None") -> None:
    """Under ``LTX_EXPORT_LOSSLESS=1`` also write the raw waveform next to the mp4 (the AAC track is lossy)."""
    if audio is None or os.environ.get("LTX_EXPORT_LOSSLESS", "0") == "0":
        return
    import numpy as np

    np.save(output_path + ".audio.npy", audio.waveform.detach().float().cpu().numpy())


def export_video_audio_yuv(yuv_planar, output_path: str, fps: int = 24, audio: Audio | None = None) -> None:
    """Export a pre-computed yuv420p planar video (+ optional audio) to MP4 — fast-path
    counterpart to :func:`export_video_audio`.

    The RGB->YUV 4:2:0 conversion already ran on-device (``fast_device_to_host_yuv``), so we
    hand libx264 native yuv420p frames and skip both the host-side color conversion and the
    3x-larger float/RGB device->host gather. The encoded MP4 is yuv420p either way, so this is
    not a fidelity change — only the conversion's location moves.

    Args:
        yuv_planar: ``(T, H*3//2, W)`` uint8 — PyAV yuv420p ndarray layout, one entry per frame.
        output_path: output .mp4 path
        fps: frame rate
        audio: decoded ``Audio``, or None
    """
    rate = audio.sampling_rate if audio is not None else None
    YuvVideoExport(yuv_planar, output_path, fps=fps, audio_sampling_rate=rate).finish(audio)


class YuvVideoExport:
    """:func:`export_video_audio_yuv` with the video track encoded on a worker thread, so the caller can
    produce the audio (e.g. decode it on device) while libx264 runs; :meth:`finish` then muxes the audio.

    The audio stream is declared up front from its sampling rate because PyAV forbids adding streams once
    muxing starts. Streams are added and packets muxed in the same order as a serial export, so the file is
    byte-identical to one. PyAV encodes with the GIL released, so the worker does not stall the caller.

    ``yuv_planar`` is read until :meth:`finish` returns; the fast YUV gather reuses its output buffer across
    calls, so the caller must not run another decode before then. It may also be a deferred array (with
    ``shape`` and ``result()``), which the worker resolves before encoding.
    """

    def __init__(self, yuv_planar, output_path: str, fps: int = 24, audio_sampling_rate: int | None = None) -> None:
        import av

        t, h32, width = yuv_planar.shape
        self._output_path = output_path
        self._fps = fps
        self._frames = t
        self._audio_sampling_rate = audio_sampling_rate

        self._container = av.open(output_path, mode="w")
        stream = self._container.add_stream("libx264", rate=int(fps))
        stream.width = width
        stream.height = h32 * 2 // 3
        stream.pix_fmt = "yuv420p"
        stream.options = _x264_options()
        stream.thread_type = "AUTO"
        self._audio_stream = None
        if audio_sampling_rate is not None:
            self._audio_stream = _add_audio_stream(self._container, Audio(torch.empty(0), audio_sampling_rate))

        # Every stream is opened and the header written here, so the audio encode in ``finish`` touches only
        # its own codec context while the worker may still be muxing video.
        self._container.start_encoding()

        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._encode_video, args=(stream, yuv_planar), name="yuv-video-export", daemon=True
        )
        self._thread.start()

    def _encode_video(self, stream, yuv_planar) -> None:
        import av

        try:
            if hasattr(yuv_planar, "result"):
                yuv_planar = yuv_planar.result()
            # Wrap each frame in place: a copy per frame (~0.45 GB per clip) is as slow as the ultrafast encode itself.
            for frame_array in yuv_planar:
                frame = av.VideoFrame.from_numpy_buffer(np.ascontiguousarray(frame_array), format="yuv420p")
                for packet in stream.encode(frame):
                    self._container.mux(packet)
            for packet in stream.encode():
                self._container.mux(packet)
        except BaseException as e:  # re-raised on the caller's thread by finish()
            self._error = e

    def finish(self, audio: Audio | None) -> None:
        """Wait for the video track, mux ``audio`` and close the file. Always closes, also on error."""
        try:
            mismatch = (audio is None) != (self._audio_stream is None) or (
                audio is not None and audio.sampling_rate != self._audio_sampling_rate
            )
            # The AAC encode (~0.15 s for 6 s of audio) overlaps the tail of the video encode; its packets are
            # muxed only after the worker is done, in the same order as a serial export.
            audio_packets = _encode_audio(self._audio_stream, audio) if audio is not None and not mismatch else []
            self._thread.join()
            if self._error is not None:
                raise self._error
            if mismatch:
                msg = f"audio {audio!r} does not match the declared rate {self._audio_sampling_rate}"
                raise ValueError(msg)
            for packet in audio_packets:
                self._container.mux(packet)
        finally:
            self._thread.join()
            self._container.close()
        _dump_audio_sidecar(self._output_path, audio)
        logger.info(f"Saved: {self._output_path} ({self._frames}f @ {self._fps}fps, yuv420p fast path)")


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
    stream.options = _x264_options()
    stream.thread_type = "AUTO"

    # Prepare audio stream if provided (must be added before any packet is muxed)
    audio_stream = _add_audio_stream(container, audio)

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
        _mux_audio(container, audio_stream, audio)

    container.close()
    _dump_audio_sidecar(output_path, audio)
    logger.info(f"Saved: {output_path} ({frames.shape[0]}f @ {fps}fps)")
