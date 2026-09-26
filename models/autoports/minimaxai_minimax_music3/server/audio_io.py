"""Waveform encoders: float32 stereo [2, N] at 44.1 kHz -> wav/flac/mp3/pcm bytes, optionally resampled to 32 kHz
(the reference server's output rate). soundfile + soxr only."""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf

CONTENT_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg", "pcm": "audio/pcm"}
SF_FORMATS = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "mp3": ("MP3", None)}


def resample(stereo: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """stereo [2, N] -> [2, M]."""
    if sr_in == sr_out:
        return stereo
    import soxr

    return np.ascontiguousarray(
        soxr.resample(np.ascontiguousarray(stereo.T), sr_in, sr_out, quality="HQ").T, dtype=np.float32
    )


def pcm16(stereo: np.ndarray) -> bytes:
    return (np.clip(stereo.T, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def encode(stereo: np.ndarray, fmt: str, sample_rate: int) -> Tuple[bytes, str]:
    if fmt == "pcm":
        return pcm16(stereo), CONTENT_TYPES["pcm"]
    if fmt not in SF_FORMATS:
        raise ValueError(f"unsupported format {fmt}")
    sff, subtype = SF_FORMATS[fmt]
    buf = io.BytesIO()
    kwargs = {"subtype": subtype} if subtype else {}
    sf.write(buf, np.clip(stereo.T, -1.0, 1.0), sample_rate, format=sff, **kwargs)
    return buf.getvalue(), CONTENT_TYPES[fmt]


def available_formats() -> dict:
    fm = sf.available_formats()
    return {k: (k == "pcm") or (SF_FORMATS.get(k, ("",))[0] in fm) for k in CONTENT_TYPES}
