# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small End-to-End Pipeline: text to .wav on Tenstorrent hardware.

Validates the complete pipeline with multiple test cases:
  1. English plain text
  2. Multilingual Spanish
  3. Emotion annotations ([laughs], [sighs])

Usage:
    python models/demos/wormhole/bark/tests/run_bark_e2e.py
    python models/demos/wormhole/bark/tests/run_bark_e2e.py --text "Custom text"
"""

import argparse
import time
from pathlib import Path

import numpy as np

import ttnn

_OUTPUT_ROOT = Path.cwd() / "bark_e2e_outputs"
_OUTPUT_FILES = {
    "english": _OUTPUT_ROOT / "bark_english.wav",
    "spanish": _OUTPUT_ROOT / "bark_spanish.wav",
    "emotion": _OUTPUT_ROOT / "bark_emotion.wav",
    "long": _OUTPUT_ROOT / "bark_long.wav",
    "custom": _OUTPUT_ROOT / "bark_output.wav",
}


def save_audio(audio: np.ndarray, output_key: str, sample_rate: int = 24000):
    """Save audio to a whitelisted WAV artifact path."""
    try:
        from scipy.io import wavfile

        if output_key not in _OUTPUT_FILES:
            raise ValueError(f"unsupported Bark E2E output artifact: {output_key}")
        output_path = _OUTPUT_FILES[output_key]
        _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        audio = np.asarray(audio, dtype=np.float32)
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16)
        print(f"  Saved: {output_path} ({len(audio_int16) / sample_rate:.2f}s)")
    except ImportError:
        print("  scipy not installed; skipping WAV save.")


def run_single(model, text: str, output_key: str, verbose: bool = True):
    """Run one text-to-audio pipeline, return timing dict."""
    timings = {}

    # Stage 1: Semantic
    t0 = time.time()
    semantic_tokens = model.generate_semantic_tokens(text)
    timings["semantic_time"] = time.time() - t0
    timings["semantic_tokens"] = semantic_tokens.shape[-1]
    timings["semantic_tps"] = timings["semantic_tokens"] / max(timings["semantic_time"], 1e-6)

    # Stage 2: Coarse
    t0 = time.time()
    coarse_tokens = model.generate_coarse_tokens(semantic_tokens)
    timings["coarse_time"] = time.time() - t0
    timings["coarse_tokens"] = coarse_tokens.shape[-1]
    timings["coarse_tps"] = timings["coarse_tokens"] / max(timings["coarse_time"], 1e-6)

    # Stage 3: Fine
    t0 = time.time()
    fine_tokens = model.generate_fine_tokens(coarse_tokens)
    timings["fine_time"] = time.time() - t0
    # Fine generates 6 new codebooks × coarse_seq_len tokens
    coarse_seq_len = timings["coarse_tokens"] // 2
    timings["fine_tokens"] = 6 * coarse_seq_len
    timings["fine_tps"] = timings["fine_tokens"] / max(timings["fine_time"], 1e-6)

    # Stage 4: Decode audio
    t0 = time.time()
    audio = model.decode_audio(fine_tokens)
    timings["decode_time"] = time.time() - t0

    # Clip and save
    audio = np.clip(audio, -1.0, 1.0)
    timings["total_time"] = (
        timings["semantic_time"] + timings["coarse_time"] + timings["fine_time"] + timings["decode_time"]
    )
    timings["audio_duration"] = len(audio) / 24000
    timings["rtf"] = timings["total_time"] / max(timings["audio_duration"], 1e-6)

    save_audio(audio, output_key)

    if verbose:
        print(
            f"  Semantic: {timings['semantic_tokens']} tokens in {timings['semantic_time']:.2f}s "
            f"({timings['semantic_tps']:.1f} tok/s)"
        )
        print(
            f"  Coarse:   {timings['coarse_tokens']} tokens in {timings['coarse_time']:.2f}s "
            f"({timings['coarse_tps']:.1f} tok/s)"
        )
        print(
            f"  Fine:     {timings['fine_tokens']} tokens in {timings['fine_time']:.2f}s "
            f"({timings['fine_tps']:.1f} tok/s)"
        )
        print(f"  Decode:   {timings['decode_time']:.2f}s")
        print(f"  Audio:    {timings['audio_duration']:.2f}s @ 24kHz")
        print(f"  RTF:      {timings['rtf']:.3f}")

    return audio, timings


def run_e2e_tests(output_dir=None):
    """Run all standard e2e test cases.

    output_dir is accepted for compatibility but intentionally ignored. WAV
    artifacts are written only to whitelisted paths under bark_e2e_outputs.
    """
    del output_dir

    from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = ttnn.open_device(device_id=0)

    try:
        print("Loading Bark Small model...")
        model = TtBarkModel(device, model_name="suno/bark-small")

        test_cases = [
            ("english", "Hello, my dog is cooler than you!"),
            ("spanish", "Hola, mi perro es mas genial que tu!"),
            ("emotion", "Hello [laughs] this is amazing [sighs] really incredible"),
            ("long", "The quick brown fox jumps over the lazy dog. " * 6),
        ]

        results = []
        for name, text in test_cases:
            print(f"\n{'='*60}")
            print(f"Test: {name}")
            print(f"Text: {text!r}")
            print(f"{'='*60}")
            audio, timings = run_single(model, text, name)
            timings["name"] = name
            timings["text"] = text
            results.append(timings)

        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Test':<12} {'Sem tok/s':>10} {'Coarse tok/s':>14} {'Audio (s)':>10} {'RTF':>8} {'Status':>8}")
        print("-" * 80)

        all_pass = True
        for r in results:
            sem_ok = r["semantic_tps"] >= 20
            coarse_ok = r["coarse_tps"] >= 60
            rtf_ok = r["rtf"] < 0.8
            status = "PASS" if (sem_ok and coarse_ok and rtf_ok) else "WARN"
            if not (sem_ok and coarse_ok and rtf_ok):
                all_pass = False
            print(
                f"{r['name']:<12} {r['semantic_tps']:>10.1f} {r['coarse_tps']:>14.1f} "
                f"{r['audio_duration']:>10.2f} {r['rtf']:>8.3f} {status:>8}"
            )

        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME WARNINGS'}")

    finally:
        ttnn.close_device(device)

    return results


def main():
    parser = argparse.ArgumentParser(description="Bark Small E2E Pipeline Test")
    parser.add_argument("--text", type=str, default=None, help="Custom text (runs single test)")
    parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.text:
        # Single custom test
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        device = ttnn.open_device(device_id=0)
        try:
            model = TtBarkModel(device, model_name="suno/bark-small")
            print(f"Input: {args.text!r}")
            run_single(model, args.text, "custom")
        finally:
            ttnn.close_device(device)
    else:
        # Full suite
        run_e2e_tests(args.output_dir)


if __name__ == "__main__":
    main()
