#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run LLM (8B) + Whisper + SpeechT5 sequentially on one shared N300 mesh device.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


def make_wav(path, duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sf.write(path, (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32), sr)


def main():
    # Force 8B for this run
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"

    logger.info("Opening shared N300 mesh device (1,2)")
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        l1_small_size=24_576,
        trace_region_size=100_000_000,
        num_command_queues=2,
    )
    mesh.enable_program_cache()

    try:
        logger.info("[1/3] Loading/running LLM 8B on shared mesh")
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

        llm = LLMTool(mesh_device=mesh)
        llm_out = llm.generate_response(
            messages=[{"role": "user", "content": "Respond with exactly: OK"}],
            max_new_tokens=16,
        )
        logger.info(f"LLM output: {llm_out!r}")
        del llm

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = f"{tmpdir}/test.wav"
            out_wav = f"{tmpdir}/tts.wav"
            make_wav(wav)

            logger.info("[2/3] Loading/running Whisper on shared mesh")
            from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

            whisper = WhisperTool(mesh_device=mesh)
            transcript = whisper.transcribe(wav)
            logger.info(f"Whisper transcript: {transcript!r}")
            whisper.close()
            del whisper

            logger.info("[3/3] Loading/running SpeechT5 on shared mesh")
            chip0 = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

            tts = SpeechT5Tool(mesh_device=chip0)
            path = tts.synthesize("Testing shared-device run.", out_wav)
            audio, sr = sf.read(path)
            logger.info(f"SpeechT5 audio: {len(audio)/sr:.2f}s @ {sr}Hz")
            del tts

        logger.info("PASS: LLM 8B + Whisper + SpeechT5 on same opened mesh")

    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
