# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small Text-to-Audio Demo.

Generates speech audio from text using the Bark Small model
running on Tenstorrent Wormhole hardware via TTNN APIs.

Usage:
    pytest models/demos/wormhole/bark/demo/demo.py -v
    # Or standalone:
    python models/demos/wormhole/bark/demo/demo.py
"""

import argparse
import time

import numpy as np


def save_audio(audio: np.ndarray, filename: str, sample_rate: int = 24000):
    """Save audio to WAV file using scipy."""
    try:
        from scipy.io import wavfile

        # Ensure NumPy array and clip to valid range before converting to int16
        audio = np.asarray(audio, dtype=np.float32)
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
        print(f"Audio saved to {filename}")
    except ImportError:
        print("scipy not installed — skipping WAV save. Install with: pip install scipy")


def run_demo(text: str = None, output_file: str = "bark_output.wav", verbose: bool = True):
    """Run the Bark demo pipeline.

    Args:
        text: Input text (default: sample sentence)
        output_file: Output WAV file path
        verbose: Print progress info
    """
    import ttnn
    from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

    if text is None:
        text = "Hello! My name is Bark and I'm running on Tenstorrent hardware. Isn't that cool?"

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        if verbose:
            print(f"Input: {text!r}")
            print()

        # Load model
        model = TtBarkModel(device, model_name="suno/bark-small")

        # Generate audio
        t0 = time.time()
        audio = model.generate(text, verbose=verbose)
        total_time = time.time() - t0

        # Clip audio to valid range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        # Save output
        save_audio(audio, output_file)

        if verbose:
            duration = len(audio) / 24000
            print(f"\n--- Summary ---")
            print(f"Input:    {text!r}")
            print(f"Audio:    {duration:.2f}s at 24kHz")
            print(f"Time:     {total_time:.2f}s")
            print(f"RTF:      {total_time / duration:.2f}")
            print(f"Output:   {output_file}")

    finally:
        ttnn.close_device(device)

    return audio


def main():
    parser = argparse.ArgumentParser(description="Bark Small Text-to-Audio Demo on Tenstorrent")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="bark_output.wav", help="Output WAV file")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    run_demo(text=args.text, output_file=args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
