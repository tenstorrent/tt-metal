#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test: Run all models on N300 with optimal mesh configuration.

- OWL-ViT, SpeechT5, BERT, Whisper: Full (1,2) mesh
- LLM: chip0 submesh (hangs on full mesh during prefill warmup)
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def make_wav(path, duration=2.0, sr=16000):
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sf.write(path, (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32), sr)


def make_img(path):
    d = np.zeros((512, 512, 3), dtype=np.uint8)
    d[40:200, 40:200] = [220, 30, 30]
    Image.fromarray(d, "RGB").save(path)


def main():
    parser = argparse.ArgumentParser(description="Test all models on full (1,2) mesh")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM (requires HF auth)")
    args = parser.parse_args()

    num_tests = 4 if args.skip_llm else 5
    logger.info("=" * 60)
    logger.info(f"TEST: {num_tests} models (4 full mesh + LLM chip0 submesh)")
    logger.info("=" * 60)

    mesh = open_n300_device()
    logger.info(f"Mesh device: {mesh.get_num_devices()} chips")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/test.png"
            wav_path = f"{tmpdir}/test.wav"
            tts_wav = f"{tmpdir}/tts.wav"
            make_img(img_path)
            make_wav(wav_path)

            # Test 1: OWL-ViT on full mesh
            logger.info(f"[1/{num_tests}] Testing OWL-ViT on full (1,2) mesh...")
            try:
                from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

                owlvit = OWLViTTool(mesh_device=mesh)  # Full mesh, not chip0
                result = owlvit.detect(img_path, "red block")
                logger.info(f"OWL-ViT on full mesh: SUCCESS - {len(result)} detections")
            except Exception as e:
                logger.error(f"OWL-ViT on full mesh: FAILED - {e}")

            # Test 2: SpeechT5 on full mesh
            logger.info(f"[2/{num_tests}] Testing SpeechT5 on full (1,2) mesh...")
            try:
                from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

                speecht5 = SpeechT5Tool(mesh_device=mesh, warmup_on_init=False)  # Full mesh
                speecht5._warmup()
                result = speecht5.synthesize("Test.", tts_wav)
                logger.info(f"SpeechT5 on full mesh: SUCCESS - {result}")
            except Exception as e:
                logger.error(f"SpeechT5 on full mesh: FAILED - {e}")

            # Test 3: BERT on full mesh
            logger.info(f"[3/{num_tests}] Testing BERT on full (1,2) mesh...")
            try:
                from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

                bert = BERTTool(mesh_device=mesh)  # Full mesh
                result = bert.qa("What?", "Two chips.")
                logger.info(f"BERT on full mesh: SUCCESS - {result!r}")
            except Exception as e:
                logger.error(f"BERT on full mesh: FAILED - {e}")

            # Test 4: Whisper on full mesh
            logger.info(f"[4/{num_tests}] Testing Whisper on full (1,2) mesh...")
            try:
                from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

                whisper = WhisperTool(mesh_device=mesh)  # Full mesh
                result = whisper.transcribe(wav_path)
                logger.info(f"Whisper on full mesh: SUCCESS - {result!r}")
                whisper.release_decoder_trace()
            except Exception as e:
                logger.error(f"Whisper on full mesh: FAILED - {e}")

            # Test 5: LLM on full (1,2) mesh with fabric enabled
            # NOTE: LLM uses full mesh for multi-chip parallelism (8B model shards across 2 chips)
            if not args.skip_llm:
                logger.info(f"[5/{num_tests}] Testing LLM on full (1,2) mesh...")
                try:
                    from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

                    llm = LLMTool(mesh_device=mesh)  # Full mesh with fabric enabled
                    result = llm.generate_response(
                        messages=[{"role": "user", "content": "Say OK"}],
                        max_new_tokens=8,
                    )
                    logger.info(f"LLM on full mesh: SUCCESS - {result!r}")
                except Exception as e:
                    logger.error(f"LLM on full mesh: FAILED - {e}")

        logger.info("=" * 60)
        logger.info("Test complete")
        logger.info("=" * 60)

    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
