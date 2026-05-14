# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small Text-to-Audio Demo.

Generates speech audio from text using the Bark Small model
running on Tenstorrent Wormhole hardware via TTNN APIs.

Usage:
    python models/demos/wormhole/bark/demo/demo.py
    python models/demos/wormhole/bark/demo/demo.py --text "Hello!" --top-k 50

    # CI entry point:
    pytest models/demos/wormhole/bark/tests/test_bark_demo.py -v
"""

import argparse
import time
from pathlib import Path

import numpy as np
from loguru import logger

_OUTPUT_FILE = Path.cwd() / "bark_output.wav"


def save_audio(audio: np.ndarray, sample_rate: int = 24000):
    """Save audio to the demo WAV artifact path."""
    try:
        from scipy.io import wavfile

        audio = np.asarray(audio, dtype=np.float32)
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        wavfile.write(str(_OUTPUT_FILE), sample_rate, audio_int16)
        logger.info(f"Audio saved to {_OUTPUT_FILE}")
    except ImportError:
        logger.warning("scipy not installed; skipping WAV save. Install with: pip install scipy")


def run_demo(text: str = None, output_file: str = None, verbose: bool = True,
             top_k: int = 0, temperature: float = 1.0, device=None):
    """Run the Bark demo pipeline.

    When called from CLI, manages its own device lifecycle.
    When called from pytest, accepts an external ``device`` from the fixture.

    Args:
        text: Text to synthesize (default: greeting)
        output_file: Ignored (for API compatibility)
        verbose: Print timing info
        top_k: If > 0, use top-k sampling for more natural speech variety
        temperature: Sampling temperature (only used if top_k > 0)
        device: Optional TTNN device (if None, opens/closes its own)

    Returns:
        audio: numpy array of 24kHz mono audio
    """
    del output_file

    import ttnn
    from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

    if text is None:
        text = "Hello! My name is Bark and I'm running on Tenstorrent hardware. Isn't that cool?"

    # Device management: use provided device or create our own
    owns_device = device is None
    if owns_device:
        device = ttnn.open_device(device_id=0)

    try:
        if verbose:
            logger.info(f"Input: {text!r}")

        # Load model
        model = TtBarkModel(device, model_name="suno/bark-small")

        # Generate audio
        t0 = time.time()
        audio = model.generate(text, verbose=verbose, top_k=top_k, temperature=temperature)
        total_time = time.time() - t0

        # Clip audio to valid range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        # Save output
        save_audio(audio)

        if verbose:
            duration = len(audio) / 24000
            logger.info(f"--- Summary ---")
            logger.info(f"Input:    {text!r}")
            logger.info(f"Audio:    {duration:.2f}s at 24kHz")
            logger.info(f"Time:     {total_time:.2f}s")
            logger.info(f"RTF:      {total_time / duration:.2f}")
            logger.info(f"Output:   {_OUTPUT_FILE}")

    finally:
        if owns_device:
            ttnn.close_device(device)

    return audio


def main():
    parser = argparse.ArgumentParser(description="Bark Small Text-to-Audio Demo on Tenstorrent")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (0=greedy, 50=natural variety)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    args = parser.parse_args()

    run_demo(
        text=args.text,
        output_file=args.output,
        verbose=not args.quiet,
        top_k=args.top_k,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
