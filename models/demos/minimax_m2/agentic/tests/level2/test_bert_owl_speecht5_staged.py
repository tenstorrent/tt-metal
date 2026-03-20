#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Staged test: BERT + OWL-ViT + SpeechT5 with BERT-first warmup.

Hypothesis: BERT must be warmed up BEFORE chip0 models are loaded.
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


def make_test_wav(path: str, duration: float = 2.0, sr: int = 16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(path, audio, sr)


def make_test_image(path: str):
    data = np.zeros((512, 512, 3), dtype=np.uint8)
    data[40:200, 40:200] = [220, 30, 30]
    data[312:472, 176:336] = [30, 180, 30]
    Image.fromarray(data, "RGB").save(path)


def main():
    logger.info("=" * 60)
    logger.info("Staged Test: BERT + OWL-ViT + SpeechT5 (BERT-first)")
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

            # === PHASE 1: Load and warmup BERT FIRST ===
            logger.info("[PHASE 1] Load and warmup BERT...")
            from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

            bert = BERTTool(mesh_device=mesh)

            bert_result = bert.qa("How many?", "The N300 has two chips.")
            assert isinstance(bert_result, str) and len(bert_result.strip()) > 0
            logger.info(f"BERT warmup: {bert_result!r}")
            gc.collect()

            # === PHASE 2: Load chip0 models ===
            logger.info("[PHASE 2] Load OWL-ViT and SpeechT5 on chip0...")
            from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

            owlvit = OWLViTTool(mesh_device=chip0)
            speecht5 = SpeechT5Tool(mesh_device=chip0, warmup_on_init=False)
            gc.collect()

            # === PHASE 3: Warmup OWL-ViT and SpeechT5 ===
            logger.info("[PHASE 3] Warmup OWL-ViT...")
            owl_result = owlvit.detect(img_path, "red block")
            assert isinstance(owl_result, list)
            logger.info(f"OWL-ViT warmup: {len(owl_result)} detections")

            logger.info("[PHASE 3] Warmup SpeechT5...")
            speecht5._warmup()
            tts_result = speecht5.synthesize("Test.", tts_wav)
            audio, sr = sf.read(tts_result)
            assert len(audio) > 0
            logger.info(f"SpeechT5 warmup: {len(audio)/sr:.2f}s")

            # === PHASE 4: Inference all ===
            logger.info("[PHASE 4] Inference all models...")
            bert2 = bert.qa("What chips?", "Two Wormhole B0 chips.")
            logger.info(f"BERT infer: {bert2!r}")

            owl2 = owlvit.detect(img_path, "green block")
            logger.info(f"OWL-ViT infer: {len(owl2)} detections")

            tts2 = speecht5.synthesize("Final test.", f"{tmpdir}/final.wav")
            audio2, sr2 = sf.read(tts2)
            logger.info(f"SpeechT5 infer: {len(audio2)/sr2:.2f}s")

        logger.info("=" * 60)
        logger.info("PASS: BERT + OWL-ViT + SpeechT5 (BERT-first staged)")
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
