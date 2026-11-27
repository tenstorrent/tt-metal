# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU export for a decoded H3 chunk; not a client-ready acknowledgement.

NUT preserves rational PTS and carries FFV1 RGB8 plus float32 stereo PCM. RGB8
quantization is explicit; no codec choice silently changes the pipeline defaults.
The caller bounds the job lifetime: cancellation/deadlines here are cooperative
between codec calls, not a way to interrupt a hung native codec.
"""

import hashlib
import io
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction


class EncodingLimit(RuntimeError):
    """The requested media or encoded output exceeded its declared bound."""


class EncodingAborted(RuntimeError):
    """Cancelled or expired export; no complete chunk is returned."""


class _BoundedBytes(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit, self.size = limit, 0

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise EncodingLimit("encoded byte limit exceeded")
        count = super().write(data)
        self.size = max(self.size, self.tell())
        return count

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.tell(), 2: self.size}.get(whence)
        if base is None or not 0 <= base + offset <= self.limit:
            raise EncodingLimit("encoded seek exceeds byte limit")
        return super().seek(offset, whence)


@dataclass(frozen=True)
class EncodedChunk:
    index: int
    data: bytes
    sha256: str
    video_pts: int
    audio_pts: int
    frames: int
    samples: int
    encode_s: float


@contextmanager
def _mux(buffer):
    import av

    mux = av.open(buffer, mode="w", format="nut", options={"avoid_negative_ts": "disabled"})
    try:
        yield mux
    except BaseException:
        # PyAV can raise a second callback error while closing after a failed
        # write. Preserve the original cancellation, byte limit or codec error.
        try:
            mux.close()
        except BaseException:
            pass
        raise
    else:
        mux.close()


class NutEncoder:
    """One chunk in flight, no retained tensors, files, queue or codec cache.

    A transport must authenticate the session/chunk identity and obtain a real
    client decode/playhead receipt before returning Session's Acknowledgement.
    This object only exports bytes; its output is never such an acknowledgement.
    """

    def __init__(self, *, pixel_format, max_frames, max_width, max_height, max_samples, max_bytes, timeout_s=120):
        bounds = (max_frames, max_width, max_height, max_samples, max_bytes)
        if any(type(value) is not int or value <= 0 for value in bounds):
            raise ValueError("positive integer media and byte bounds required")
        if pixel_format != "rgb8":
            raise ValueError("explicit pixel_format='rgb8' quantization required")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("positive finite codec timeout required")
        self.bounds, self.timeout_s = bounds, timeout_s

    def __call__(self, chunk, *, cancelled=lambda: False):
        import av
        import numpy as np

        started = time.monotonic()

        def check():
            if cancelled() or time.monotonic() - started >= self.timeout_s:
                raise EncodingAborted("encoding cancelled or deadline exceeded")

        check()
        media = chunk.media
        video, audio = media.video, media.audio
        integers = (chunk.index, chunk.video_pts, chunk.audio_pts, media.frames, media.samples)
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("nonnegative integer chunk counters required")
        if (chunk.video_time_base, chunk.audio_time_base) != (Fraction(1, 24), Fraction(1, 32000)):
            raise ValueError("H3 export requires 24 fps and 32000 Hz")
        if chunk.audio_pts != chunk.video_pts * 32000 // 24:
            raise ValueError("audio and video origins do not share the stream timeline")
        if media.samples != (chunk.video_pts + media.frames) * 32000 // 24 - chunk.audio_pts:
            raise ValueError("audio length differs from the rational video interval")
        if video.device.type != "cpu" or audio.device.type != "cpu":
            raise ValueError("no implicit device copies during export")
        if not video.is_floating_point() or not audio.is_floating_point():
            raise ValueError("expected decoded floating-point media")
        if video.ndim != 4 or video.shape[:2] != (media.frames, 3) or audio.shape != (2, media.samples):
            raise ValueError("expected TCHW RGB and stereo CN audio")
        sizes = (media.frames, video.shape[3], video.shape[2], media.samples)
        if any(size <= 0 or size > bound for size, bound in zip(sizes, self.bounds)):
            raise EncodingLimit("decoded media exceeds configured shape bound")
        with _BoundedBytes(self.bounds[-1]) as buffer:
            with _mux(buffer) as mux:
                vs = mux.add_stream("ffv1", rate=24)
                vs.width, vs.height, vs.pix_fmt = video.shape[3], video.shape[2], "bgr0"
                vs.time_base = chunk.video_time_base
                vs.codec_context.thread_count = 1
                aus = mux.add_stream("pcm_f32le", rate=32000)
                aus.layout, aus.time_base = "stereo", chunk.audio_time_base
                audio_offset = 0
                for index in range(media.frames):
                    # Check outside C I/O callbacks: PyAV can swallow
                    # KeyboardInterrupt/SystemExit raised from those callbacks.
                    check()
                    pixels = video[index].detach().float().movedim(0, -1).numpy()
                    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                        raise ValueError("RGB must be finite and within [0, 1]; no implicit clipping")
                    frame = av.VideoFrame.from_ndarray((pixels * 255).round().astype("uint8"), format="rgb24")
                    frame.pts, frame.time_base = chunk.video_pts + index, chunk.video_time_base
                    for packet in vs.encode(frame):
                        mux.mux(packet)
                    # Interleave by rational global boundaries; never round each
                    # chunk's sound independently or queue all video before audio.
                    stop = (chunk.video_pts + index + 1) * 32000 // 24 - chunk.audio_pts
                    samples = audio[:, audio_offset:stop].detach().float().T.contiguous().numpy()
                    if not np.isfinite(samples).all():
                        raise ValueError("PCM samples must be finite")
                    sound = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="stereo")
                    sound.sample_rate = 32000
                    sound.pts, sound.time_base = chunk.audio_pts + audio_offset, chunk.audio_time_base
                    for packet in aus.encode(sound):
                        mux.mux(packet)
                    audio_offset = stop
                for stream in (vs, aus):
                    for packet in stream.encode(None):
                        mux.mux(packet)
            check()
            data = buffer.getvalue()
        return EncodedChunk(
            chunk.index,
            data,
            hashlib.sha256(data).hexdigest(),
            chunk.video_pts,
            chunk.audio_pts,
            media.frames,
            media.samples,
            time.monotonic() - started,
        )
