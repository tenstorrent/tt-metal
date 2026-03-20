#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test: Run ALL non-trace models on chip0 submesh.

Hypothesis: If BERT runs on chip0 (instead of full mesh), it won't conflict with SpeechT5.
"""
import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def make_test_image(path: str):
    data = np.zeros((512, 512, 3), dtype=np.uint8)
    data[40:200, 40:200] = [220, 30, 30]
    Image.fromarray(data, "RGB").save(path)


def main():
    logger.info("=" * 60)
    logger.info("Test: ALL models on chip0 submesh")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        chip0 = (
            mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)) if mesh.get_num_devices() > 1 else mesh
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/test.png"
            tts_wav = f"{tmpdir}/tts.wav"
            make_test_image(img_path)

            # Load ALL models on chip0
            logger.info("[LOAD] BERT on chip0...")
            from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

            bert = BERTTool(mesh_device=chip0)  # chip0 instead of mesh

            logger.info("[LOAD] OWL-ViT on chip0...")
            from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

            owlvit = OWLViTTool(mesh_device=chip0)

            logger.info("[LOAD] SpeechT5 on chip0...")
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

            speecht5 = SpeechT5Tool(mesh_device=chip0, warmup_on_init=False)

            gc.collect()
            logger.info("All models loaded on chip0.")

            # Warmup all
            logger.info("[WARMUP] BERT...")
            bert_result = bert.qa("How many?", "Two chips.")
            logger.info(f"BERT: {bert_result!r}")

            logger.info("[WARMUP] OWL-ViT...")
            owl_result = owlvit.detect(img_path, "block")
            logger.info(f"OWL-ViT: {len(owl_result)} detections")

            logger.info("[WARMUP] SpeechT5...")
            speecht5._warmup()
            tts_result = speecht5.synthesize("Test.", tts_wav)
            audio, sr = sf.read(tts_result)
            logger.info(f"SpeechT5: {len(audio)/sr:.2f}s")

            # Inference all
            logger.info("[INFER] All models...")
            bert2 = bert.qa("What?", "Wormhole B0.")
            owl2 = owlvit.detect(img_path, "red")
            tts2 = speecht5.synthesize("Done.", f"{tmpdir}/done.wav")
            audio2, _ = sf.read(tts2)

            logger.info(f"BERT: {bert2!r}")
            logger.info(f"OWL: {len(owl2)} detections")
            logger.info(f"TTS: {len(audio2)/16000:.2f}s")

        logger.info("=" * 60)
        logger.info("PASS: All on chip0")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
