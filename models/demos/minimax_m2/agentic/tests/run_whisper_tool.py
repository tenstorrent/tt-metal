#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone test: WhisperTool on N300 mesh device."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from loguru import logger

import ttnn


def make_wav(path, duration=2.0, sr=16000):
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(path, audio, sr)
    return path


def main():
    from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import (
        WHISPER_L1_SMALL_SIZE,
        WHISPER_TRACE_REGION_SIZE,
    )

    logger.info(f"Opening N300 mesh for Whisper (l1={WHISPER_L1_SMALL_SIZE}, trace={WHISPER_TRACE_REGION_SIZE})")
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        l1_small_size=WHISPER_L1_SMALL_SIZE,
        trace_region_size=WHISPER_TRACE_REGION_SIZE,
        num_command_queues=1,
    )
    mesh_device.enable_program_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = make_wav(f"{tmpdir}/test.wav")

        whisper = None
        try:
            from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

            logger.info("Loading WhisperTool...")
            whisper = WhisperTool(mesh_device=mesh_device)

            # First call
            logger.info("Transcribe call 1 (compile + trace capture)...")
            result1 = whisper.transcribe(wav)
            logger.info(f"Transcription: {result1!r}")

            # Second call (trace reuse)
            logger.info("Transcribe call 2 (trace reuse)...")
            result2 = whisper.transcribe(wav)
            logger.info(f"Transcription (2nd): {result2!r}")

            # Translate
            logger.info("Translate call...")
            result3 = whisper.translate(wav)
            logger.info(f"Translation: {result3!r}")

            assert isinstance(result1, str), "transcribe must return str"
            assert isinstance(result2, str), "2nd transcribe must return str"
            assert isinstance(result3, str), "translate must return str"

            logger.info(f"All results OK. Transcription 1: {result1!r}")
            logger.info("Whisper tool: ALL DONE")
        finally:
            if whisper is not None:
                whisper.close()  # release traces before device close
            ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
