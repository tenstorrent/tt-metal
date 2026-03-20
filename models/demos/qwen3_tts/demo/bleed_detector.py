#!/usr/bin/env python
"""
Bleed Detector - Utility for detecting and measuring reference audio bleed in TTS output.

Uses Whisper word-level timestamps to detect where the target text actually starts
in the generated audio, identifying any prefix content from reference audio bleeding.
"""

from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from transformers import pipeline


def detect_bleed(
    audio_path: str,
    target_first_word: str = "Hello",
    whisper_model: str = "openai/whisper-base",
) -> Dict:
    """
    Detect reference audio bleed in TTS output.

    Args:
        audio_path: Path to the generated audio file
        target_first_word: The expected first word of the target text (case-insensitive)
        whisper_model: Whisper model to use for transcription

    Returns:
        Dict with:
            - transcription: Full transcription
            - word_timestamps: List of (start, end, word) tuples
            - target_word_start: Time when target_first_word starts (seconds)
            - bleed_duration: Amount of bleed in seconds (0 if no bleed)
            - bleed_content: Transcribed content before target word
            - recommended_trim_frames: Recommended trim value in codec frames (12Hz)
    """
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr

    # Run Whisper
    pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        return_timestamps="word",
    )
    result = pipe(audio_path)

    # Extract word timestamps
    word_timestamps = []
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        word = chunk["text"].strip()
        word_timestamps.append((start, end, word))

    # Find target word
    target_start = None
    target_idx = None
    target_lower = target_first_word.lower().rstrip(",.")

    for i, (start, end, word) in enumerate(word_timestamps):
        word_clean = word.lower().rstrip(",.")
        if word_clean == target_lower or target_lower in word_clean:
            target_start = start
            target_idx = i
            break

    # Calculate bleed
    bleed_duration = target_start if target_start is not None else 0.0
    bleed_content = ""
    if target_idx is not None and target_idx > 0:
        bleed_content = " ".join(word for _, _, word in word_timestamps[:target_idx])

    # Calculate recommended trim frames (12Hz codec = 83.3ms per frame)
    # Add small margin to ensure complete removal
    frames_per_second = 12
    recommended_trim = int(np.ceil(bleed_duration * frames_per_second)) + 2

    return {
        "audio_path": audio_path,
        "audio_duration": duration,
        "transcription": result["text"],
        "word_timestamps": word_timestamps,
        "target_word": target_first_word,
        "target_word_start": target_start,
        "bleed_duration": bleed_duration,
        "bleed_content": bleed_content,
        "recommended_trim_frames": recommended_trim if bleed_duration > 0.1 else 0,
    }


def trim_audio(
    audio_path: str,
    output_path: str,
    trim_seconds: float,
) -> str:
    """
    Trim the beginning of an audio file.

    Args:
        audio_path: Input audio path
        output_path: Output audio path
        trim_seconds: Seconds to trim from start

    Returns:
        Output path
    """
    audio, sr = sf.read(audio_path)
    trim_samples = int(trim_seconds * sr)
    trimmed = audio[trim_samples:]
    sf.write(output_path, trimmed, sr)
    return output_path


def auto_trim_bleed(
    audio_path: str,
    output_path: str,
    target_first_word: str = "Hello",
    margin_seconds: float = 0.1,
) -> Tuple[str, Dict]:
    """
    Automatically detect and trim reference bleed from TTS output.

    Args:
        audio_path: Input audio path
        output_path: Output audio path
        target_first_word: Expected first word of target text
        margin_seconds: Extra margin to trim before target word

    Returns:
        Tuple of (output_path, detection_results)
    """
    # Detect bleed
    results = detect_bleed(audio_path, target_first_word)

    bleed = results["bleed_duration"]
    if bleed > 0.1:
        # Trim with small margin
        trim_time = max(0, bleed - margin_seconds)
        trim_audio(audio_path, output_path, trim_time)
        results["trimmed"] = True
        results["trim_seconds"] = trim_time
    else:
        # No significant bleed, just copy
        import shutil

        shutil.copy(audio_path, output_path)
        results["trimmed"] = False
        results["trim_seconds"] = 0

    return output_path, results


def print_bleed_report(results: Dict):
    """Print a formatted bleed detection report."""
    print(f"\n{'='*60}")
    print("BLEED DETECTION REPORT")
    print(f"{'='*60}")
    print(f"Audio: {results['audio_path']}")
    print(f"Duration: {results['audio_duration']:.2f}s")
    print(f"Target word: '{results['target_word']}'")
    print(f"\nTranscription: {results['transcription']}")

    print(f"\nWord timestamps (first 10):")
    for start, end, word in results["word_timestamps"][:10]:
        marker = " <-- TARGET" if word.lower().rstrip(",.") == results["target_word"].lower() else ""
        print(f"  {start:.2f}s - {end:.2f}s: '{word}'{marker}")

    print(f"\n{'='*60}")
    if results["bleed_duration"] > 0.1:
        print(f"BLEED DETECTED: {results['bleed_duration']:.2f}s")
        print(f"Bleed content: '{results['bleed_content']}'")
        print(f"Recommended trim: {results['recommended_trim_frames']} codec frames")
    else:
        print("NO SIGNIFICANT BLEED DETECTED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect reference audio bleed in TTS output")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--target-word", default="Hello", help="Expected first word of target text")
    parser.add_argument("--auto-trim", help="Auto-trim and save to this path")
    parser.add_argument("--whisper-model", default="openai/whisper-base", help="Whisper model")

    args = parser.parse_args()

    if args.auto_trim:
        output, results = auto_trim_bleed(
            args.audio,
            args.auto_trim,
            args.target_word,
        )
        print_bleed_report(results)
        if results["trimmed"]:
            print(f"Trimmed audio saved to: {output}")
    else:
        results = detect_bleed(args.audio, args.target_word, args.whisper_model)
        print_bleed_report(results)
