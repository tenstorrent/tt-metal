# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Long text support for Bark Small.

Bark works best with sentences under ~250 characters. This module splits
longer inputs into chunks, generates audio per chunk, and concatenates
the segments with a short crossfade to avoid audible seams.

Usage:
    from models.demos.wormhole.bark.tt.bark_long_text import generate_long_text_audio

    audio = generate_long_text_audio("Very long text...", bark_model)
"""

import re

import numpy as np


def split_long_text(text: str, max_chars: int = 250) -> list:
    """Split text into chunks suitable for Bark generation.

    Splits on sentence boundaries (. ! ?) first, then on commas/semicolons,
    and finally on word boundaries as a last resort.

    Args:
        text: Input text of any length.
        max_chars: Maximum characters per chunk (default 250).

    Returns:
        List of text chunks, each <= max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(sentence) > max_chars:
                sub_chunks = _split_sentence(sentence, max_chars)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_sentence(sentence: str, max_chars: int) -> list:
    """Split a single long sentence on commas/semicolons or word boundaries."""
    parts = re.split(r"[,;]\s*", sentence)
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) + 2 <= max_chars:
            current = f"{current}, {part}".strip(", ")
        else:
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                words = part.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current = f"{current} {word}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


def crossfade_segments(segments: list, crossfade_ms: int = 50, sample_rate: int = 24000) -> np.ndarray:
    """Concatenate audio segments with a short crossfade.

    Args:
        segments: List of numpy float32 audio arrays (values in [-1, 1]).
        crossfade_ms: Crossfade duration in milliseconds (default 50).
        sample_rate: Audio sample rate (default 24000).

    Returns:
        Concatenated audio as a single numpy array.
    """
    if len(segments) == 0:
        return np.array([], dtype=np.float32)
    if len(segments) == 1:
        return segments[0]

    crossfade_samples = int(crossfade_ms * sample_rate / 1000)
    result = segments[0].copy()

    for seg in segments[1:]:
        seg = seg.copy()
        if crossfade_samples > 0 and len(result) > crossfade_samples and len(seg) > crossfade_samples:
            fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

            result[-crossfade_samples:] *= fade_out
            seg[:crossfade_samples] *= fade_in
            result[-crossfade_samples:] += seg[:crossfade_samples]
            result = np.concatenate([result, seg[crossfade_samples:]])
        else:
            silence = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
            result = np.concatenate([result, silence, seg])

    return result


def generate_long_text_audio(
    text: str,
    bark_model,
    crossfade_ms: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Generate audio for text of any length (500+ characters supported).

    Splits into chunks, generates audio per chunk via the TtBarkModel pipeline,
    and concatenates with crossfade.

    Args:
        text: Input text (any length).
        bark_model: A TtBarkModel instance.
        crossfade_ms: Crossfade duration in milliseconds.
        verbose: Print per-chunk progress.

    Returns:
        numpy int16 audio array at 24 kHz.
    """
    chunks = split_long_text(text, max_chars=250)
    if verbose:
        print(f"Long text: split into {len(chunks)} chunk(s)")

    audio_segments = []
    for i, chunk in enumerate(chunks):
        if verbose:
            display = chunk[:60] + "..." if len(chunk) > 60 else chunk
            print(f"  Chunk [{i + 1}/{len(chunks)}]: '{display}'")

        audio = bark_model.generate(chunk, verbose=False)
        # Ensure float32 for crossfade math
        audio_f32 = np.asarray(audio, dtype=np.float32)
        if np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0  # Handle int16 input
        audio_segments.append(audio_f32)

    final = crossfade_segments(audio_segments, crossfade_ms=crossfade_ms)
    final = np.clip(final, -1.0, 1.0)

    if verbose:
        duration = len(final) / 24000
        print(f"  Total audio: {duration:.2f}s")

    return final
