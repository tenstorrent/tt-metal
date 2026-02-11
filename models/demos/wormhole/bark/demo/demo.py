# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Demo script for Bark Small on Tenstorrent Wormhole.

Generates audio from text using the full 3-stage pipeline.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile
import torch
from loguru import logger

import ttnn

from models.demos.wormhole.bark.tt.bark_model import TtBarkModel


SAMPLE_TEXTS = [
    "Hello, my dog is cooler than you!",
    "Hey there! [laughs] This is Bark running on Tenstorrent hardware.",
    "Buenos días, ¿cómo estás hoy?",
    "The quick brown fox [sighs] jumps over the lazy dog.",
]


def run_bark_demo(
    device: ttnn.Device,
    texts: list = None,
    output_dir: str = "bark_outputs",
):
    """Run Bark Small demo generating audio from text."""
    if texts is None:
        texts = SAMPLE_TEXTS

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bark Small - Text-to-Audio Demo")
    logger.info("Running on Tenstorrent Wormhole")
    logger.info("=" * 60)

    # Load model
    t0 = time.time()
    model = TtBarkModel.from_pretrained(device)
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    results = []
    total_audio_duration = 0.0
    total_generation_time = 0.0

    for i, text in enumerate(texts):
        logger.info(f"\n--- Sample {i + 1}/{len(texts)} ---")
        logger.info(f"Input: '{text}'")

        t0 = time.time()
        audio = model.generate(
            text,
            max_semantic_tokens=256,
            max_coarse_tokens=256,
        )
        gen_time = time.time() - t0

        audio_duration = len(audio) / 24_000
        total_audio_duration += audio_duration
        total_generation_time += gen_time

        # Save output
        output_path = os.path.join(output_dir, f"bark_sample_{i}.wav")
        scipy.io.wavfile.write(output_path, rate=24_000, data=audio)
        logger.info(
            f"Generated {audio_duration:.2f}s audio in {gen_time:.2f}s "
            f"(RTF={gen_time / audio_duration:.2f})"
        )
        logger.info(f"Saved to: {output_path}")

        results.append(
            {
                "text": text,
                "audio_duration": audio_duration,
                "generation_time": gen_time,
                "rtf": gen_time / audio_duration,
                "output_path": output_path,
            }
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    avg_rtf = total_generation_time / total_audio_duration if total_audio_duration > 0 else float("inf")
    logger.info(f"Total audio generated: {total_audio_duration:.2f}s")
    logger.info(f"Total generation time: {total_generation_time:.2f}s")
    logger.info(f"Average RTF: {avg_rtf:.2f}")
    logger.info(f"Outputs saved to: {output_dir}/")

    return results


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize(
    "text",
    [
        "Hello, my dog is cooler than you!",
        "Hey there! [laughs] This is awesome.",
    ],
)
def test_bark_demo(device, text):
    """Run demo with a single text sample."""
    model = TtBarkModel.from_pretrained(device)
    audio = model.generate(text, max_semantic_tokens=128, max_coarse_tokens=128)

    assert audio is not None
    assert len(audio) > 0
    assert isinstance(audio, np.ndarray)

    audio_duration = len(audio) / 24_000
    logger.info(f"Generated {audio_duration:.2f}s of audio")
    assert audio_duration > 0.1, "Audio too short, something went wrong"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        run_bark_demo(device)
    finally:
        ttnn.close_device(device)
