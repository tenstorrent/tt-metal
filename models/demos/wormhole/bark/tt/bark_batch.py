# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Batch audio generation for Bark Small.

Processes multiple text inputs sequentially (batch=1 per stage)
and reports per-item and aggregate performance metrics.

Usage:
    from models.demos.wormhole.bark.tt.bark_batch import batch_generate_audio

    results = batch_generate_audio(["Hello!", "Goodbye!"], bark_model, output_dir="outputs")
"""

import os
import time

import numpy as np


def batch_generate_audio(
    texts: list,
    bark_model,
    output_dir: str = "outputs",
    voice_preset=None,
    verbose: bool = True,
) -> list:
    """Generate audio for multiple text inputs.

    Processes each text sequentially through the full Bark pipeline
    and saves individual WAV files.

    Args:
        texts: List of input text strings.
        bark_model: A TtBarkModel instance.
        output_dir: Directory to save output WAV files.
        voice_preset: Optional voice preset for consistent speaker.
        verbose: Print per-item progress.

    Returns:
        List of result dicts with keys: text, output, duration, time, rtf
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_start = time.time()

    for i, text in enumerate(texts):
        if verbose:
            display = text[:50] + "..." if len(text) > 50 else text
            print(f"\n[{i + 1}/{len(texts)}] '{display}'")

        t0 = time.time()
        audio = bark_model.generate(text, voice_preset=voice_preset, verbose=False)
        elapsed = time.time() - t0

        # Save WAV
        output_path = os.path.join(output_dir, f"output_{i:03d}.wav")
        _save_wav(audio, output_path)

        duration = len(audio) / 24000
        rtf = elapsed / duration if duration > 0 else float("inf")

        results.append(
            {
                "text": text,
                "output": output_path,
                "duration": duration,
                "time": elapsed,
                "rtf": rtf,
            }
        )

        if verbose:
            print(f"  → {duration:.2f}s audio in {elapsed:.2f}s (RTF={rtf:.3f})")

    total_time = time.time() - total_start

    if verbose and len(results) > 0:
        avg_rtf = np.mean([r["rtf"] for r in results])
        total_audio = sum(r["duration"] for r in results)
        print(f"\n{'='*50}")
        print(f"BATCH SUMMARY ({len(texts)} items)")
        print(f"  Total time:  {total_time:.2f}s")
        print(f"  Total audio: {total_audio:.2f}s")
        print(f"  Avg RTF:     {avg_rtf:.3f}")
        print(f"  Output dir:  {output_dir}")
        print(f"{'='*50}")

    return results


def _save_wav(audio: np.ndarray, filename: str, sample_rate: int = 24000):
    """Save audio array to WAV file."""
    try:
        from scipy.io import wavfile

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
    except ImportError:
        print(f"scipy not installed — cannot save {filename}")
