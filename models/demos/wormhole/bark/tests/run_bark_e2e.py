# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small End-to-End Pipeline — text → .wav on Tenstorrent hardware.

Validates the complete pipeline with multiple test cases:
  1. English plain text
  2. Multilingual (Spanish)
  3. Emotion annotations ([laughs], [sighs])

Usage:
    python models/demos/wormhole/bark/tests/run_bark_e2e.py
    python models/demos/wormhole/bark/tests/run_bark_e2e.py --text "Custom text" --output my_audio.wav
"""

import argparse
import os
import time

import numpy as np

import ttnn


def save_audio(audio: np.ndarray, filename: str, sample_rate: int = 24000):
    """Save audio to WAV file using scipy."""
    try:
        from scipy.io import wavfile

        audio = np.asarray(audio, dtype=np.float32)
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
        print(f"  ✓ Saved: {filename} ({len(audio_int16) / sample_rate:.2f}s)")
    except ImportError:
        print("  ⚠ scipy not installed — skipping WAV save.")


def run_single(model, text: str, output_file: str, verbose: bool = True):
    """Run one text → audio pipeline, return timing dict."""
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

    # Stage 4: Decode audio
    t0 = time.time()
    audio = model.decode_audio(fine_tokens)
    timings["decode_time"] = time.time() - t0

    # Clip and save
    audio = np.clip(audio, -1.0, 1.0)
    timings["total_time"] = timings["semantic_time"] + timings["coarse_time"] + timings["fine_time"] + timings["decode_time"]
    timings["audio_duration"] = len(audio) / 24000
    timings["rtf"] = timings["total_time"] / max(timings["audio_duration"], 1e-6)

    save_audio(audio, output_file)

    if verbose:
        print(f"  Semantic: {timings['semantic_tokens']} tokens in {timings['semantic_time']:.2f}s ({timings['semantic_tps']:.1f} tok/s)")
        print(f"  Coarse:   {timings['coarse_tokens']} tokens in {timings['coarse_time']:.2f}s ({timings['coarse_tps']:.1f} tok/s)")
        print(f"  Fine:     {timings['fine_time']:.2f}s")
        print(f"  Decode:   {timings['decode_time']:.2f}s")
        print(f"  Audio:    {timings['audio_duration']:.2f}s @ 24kHz")
        print(f"  RTF:      {timings['rtf']:.3f}")

    return audio, timings


def run_e2e_tests(output_dir: str = "bark_e2e_outputs"):
    """Run all standard e2e test cases."""
    from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

    os.makedirs(output_dir, exist_ok=True)
    device = ttnn.open_device(device_id=0)

    try:
        print("Loading Bark Small model...")
        model = TtBarkModel(device, model_name="suno/bark-small")

        test_cases = [
            ("english", "Hello, my dog is cooler than you!"),
            ("spanish", "Hola, mi perro es más genial que tú!"),
            ("emotion", "Hello [laughs] this is amazing [sighs] really incredible"),
            ("long", "The quick brown fox jumps over the lazy dog. " * 6),
        ]

        results = []
        for name, text in test_cases:
            output_file = os.path.join(output_dir, f"bark_{name}.wav")
            print(f"\n{'='*60}")
            print(f"Test: {name}")
            print(f"Text: {text!r}")
            print(f"{'='*60}")
            audio, timings = run_single(model, text, output_file)
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
            rtf_ok = r["rtf"] < 0.8
            status = "PASS" if (sem_ok and rtf_ok) else "WARN"
            if not (sem_ok and rtf_ok):
                all_pass = False
            print(f"{r['name']:<12} {r['semantic_tps']:>10.1f} {r['coarse_tps']:>14.1f} {r['audio_duration']:>10.2f} {r['rtf']:>8.3f} {status:>8}")

        print(f"\nOverall: {'ALL PASS ✓' if all_pass else 'SOME WARNINGS ⚠'}")

    finally:
        ttnn.close_device(device)

    return results


def main():
    parser = argparse.ArgumentParser(description="Bark Small E2E Pipeline Test")
    parser.add_argument("--text", type=str, default=None, help="Custom text (runs single test)")
    parser.add_argument("--output", type=str, default="bark_output.wav", help="Output WAV file")
    parser.add_argument("--output-dir", type=str, default="bark_e2e_outputs", help="Output directory for batch tests")
    args = parser.parse_args()

    if args.text:
        # Single custom test
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        device = ttnn.open_device(device_id=0)
        try:
            model = TtBarkModel(device, model_name="suno/bark-small")
            print(f"Input: {args.text!r}")
            run_single(model, args.text, args.output)
        finally:
            ttnn.close_device(device)
    else:
        # Full suite
        run_e2e_tests(args.output_dir)


if __name__ == "__main__":
    main()
