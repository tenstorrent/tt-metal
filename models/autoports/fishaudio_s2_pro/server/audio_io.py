"""Audio bytes <-> float32 waveforms, WAV streaming header, encoders. soundfile + soxr only (no torchaudio)."""
from __future__ import annotations

import io
import struct
from typing import Tuple

import numpy as np
import soundfile as sf

from models.autoports.fishaudio_s2_pro.config import SAMPLE_RATE

CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/pcm", "opus": "audio/ogg", "flac": "audio/flac"}
SF_FORMATS = {"wav": ("WAV", "PCM_16"), "mp3": ("MP3", None), "flac": ("FLAC", "PCM_16"), "opus": ("OGG", "OPUS")}


def decode_audio(data: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Any soundfile-readable container -> mono float32 at target_sr."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import soxr

        wav = soxr.resample(wav, sr, target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def wav_chunk_header(sample_rate: int = SAMPLE_RATE, bit_depth: int = 16, channels: int = 1) -> bytes:
    """fish-speech's 44-byte streaming WAV header with a zero data length."""
    byte_rate = sample_rate * channels * bit_depth // 8
    block_align = channels * bit_depth // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bit_depth)
        + b"data"
        + struct.pack("<I", 0)
    )


def pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def encode(wav: np.ndarray, fmt: str, sample_rate: int = SAMPLE_RATE) -> Tuple[bytes, str]:
    """Whole-file encode. Returns (bytes, content-type)."""
    if fmt == "pcm":
        return pcm16(wav), CONTENT_TYPES["pcm"]
    if fmt not in SF_FORMATS:
        raise ValueError(f"unsupported format {fmt}")
    sff, subtype = SF_FORMATS[fmt]
    buf = io.BytesIO()
    kwargs = {"subtype": subtype} if subtype else {}
    sf.write(buf, wav, sample_rate, format=sff, **kwargs)
    return buf.getvalue(), CONTENT_TYPES[fmt]


def available_formats() -> dict:
    return {k: (k == "pcm") or (SF_FORMATS.get(k, ("",))[0] in sf.available_formats()) for k in CONTENT_TYPES}
