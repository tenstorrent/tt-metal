#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 0 Test: Whisper standalone with shared device parameters.

Verifies Whisper can:
1. Load on N300 mesh with shared device params
2. Warmup (capture decoder trace)
3. Run inference
4. Release decoder trace cleanly
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def make_test_wav(path: str, duration: float = 2.0, sr: int = 16000):
    """Create a simple sine wave test audio file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(path, audio, sr)


def test_whisper_standalone():
    """Test Whisper in isolation with shared device params."""
    logger.info("=" * 60)
    logger.info("Level 0: Whisper Standalone Test")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = f"{tmpdir}/test.wav"
            make_test_wav(wav_path)

            # Load
            logger.info("[1/4] Loading Whisper...")
            from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

            whisper = WhisperTool(mesh_device=mesh)
            logger.info("Whisper loaded OK")

            # Warmup (trace capture)
            logger.info("[2/4] Warmup (trace capture)...")
            result = whisper.transcribe(wav_path)
            assert isinstance(result, str), f"Expected str, got {type(result)}"
            logger.info(f"Warmup result: {result!r}")

            # Inference
            logger.info("[3/4] Inference...")
            result2 = whisper.transcribe(wav_path)
            assert isinstance(result2, str), f"Expected str, got {type(result2)}"
            logger.info(f"Inference result: {result2!r}")

            # Cleanup
            logger.info("[4/4] Release decoder trace...")
            whisper.release_decoder_trace()
            whisper.close()
            logger.info("Trace released OK")

        logger.info("=" * 60)
        logger.info("PASS: Whisper standalone test")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: Whisper standalone test: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    success = test_whisper_standalone()
    sys.exit(0 if success else 1)
