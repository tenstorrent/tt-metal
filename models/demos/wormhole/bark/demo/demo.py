# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small - Command line demo script.
Generates audio from text and saves as WAV files.
"""

import os
import time
import numpy as np
import scipy.io.wavfile
from loguru import logger

import ttnn
from models.demos.wormhole.bark.tt.bark_model import TtBarkModel


def run_demo(text: str, output_file: str, device: ttnn.Device):
    """Run text-to-audio generation and save output."""
    logger.info(f"Initialising Bark Small model...")
    model = TtBarkModel.from_pretrained(device)

    logger.info(f"Generating audio for text: '{text}'")
    t0 = time.time()
    audio = model.generate(text, max_semantic_tokens=256, max_coarse_tokens=256)
    t1 = time.time()

    duration = len(audio) / 24_000
    latency = t1 - t0
    rtf = latency / duration if duration > 0 else 0

    logger.info(f"Generation complete!")
    logger.info(f"Audio duration: {duration:.2f}s")
    logger.info(f"Latency: {latency:.2f}s")
    logger.info(f"Real-Time Factor (RTF): {rtf:.2f}")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to WAV
    scipy.io.wavfile.write(output_file, 24_000, audio)
    logger.info(f"Saved audio to: {output_file}")


if __name__ == "__main__":
    # Command line entrypoint
    device = ttnn.open_device(device_id=0)

    try:
        sample_text = "Hello, my dog is cooler than you! [laughs]"
        run_demo(sample_text, "bark_out.wav", device)
    finally:
        ttnn.close_device(device)
