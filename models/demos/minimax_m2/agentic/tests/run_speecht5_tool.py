#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone test: SpeechT5Tool on single device."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from loguru import logger

import ttnn


def main():
    # SpeechT5 demo uses: l1_small_size=300000, trace_region_size=10000000, num_command_queues=2
    logger.info("Opening single device for SpeechT5...")
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=300000,
        trace_region_size=10_000_000,
        num_command_queues=2,
    )
    device.enable_program_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = f"{tmpdir}/speech.wav"
        try:
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

            logger.info("Loading SpeechT5Tool...")
            tts = SpeechT5Tool(mesh_device=device)

            # First synthesis
            logger.info("Synthesizing speech (first call)...")
            path = tts.synthesize("Hello world! This is a TTS test.", output_path=out_wav)
            import os

            import soundfile as sf

            assert os.path.exists(path), "Output file not found"
            audio, sr = sf.read(path)
            assert len(audio) > 0, "Empty audio output"
            assert sr == 16000, f"Unexpected sample rate: {sr}"
            logger.info(f"Audio: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)")

            # Second synthesis
            logger.info("Synthesizing speech (second call, kernel reuse)...")
            path2 = tts.synthesize("The quick brown fox jumps over the lazy dog.", output_path=out_wav)
            audio2, _ = sf.read(path2)
            assert len(audio2) > 0, "Empty audio on second call"
            logger.info(f"Audio2: {len(audio2)} samples ({len(audio2)/sr:.2f}s)")

            logger.info("SpeechT5 tool: ALL DONE")
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
