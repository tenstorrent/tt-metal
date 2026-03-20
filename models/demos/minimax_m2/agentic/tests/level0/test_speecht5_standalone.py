#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 0 Test: SpeechT5 standalone with shared device parameters.

Verifies SpeechT5 can:
1. Load on chip0 submesh with shared device params
2. Run warmup (KV-cache mode, no trace)
3. Run synthesis inference
"""
import sys
import tempfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def test_speecht5_standalone():
    """Test SpeechT5 in isolation with shared device params."""
    logger.info("=" * 60)
    logger.info("Level 0: SpeechT5 Standalone Test")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        # Create chip0 submesh (SpeechT5 needs single-buffer tensors)
        chip0 = (
            mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)) if mesh.get_num_devices() > 1 else mesh
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            warmup_wav = f"{tmpdir}/warmup.wav"
            infer_wav = f"{tmpdir}/infer.wav"

            # Load (warmup_on_init=False to control timing)
            logger.info("[1/4] Loading SpeechT5...")
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

            speecht5 = SpeechT5Tool(mesh_device=chip0, warmup_on_init=False)
            logger.info("SpeechT5 loaded OK")

            # Warmup
            logger.info("[2/4] Warmup...")
            speecht5._warmup()
            logger.info("Warmup complete")

            # Warmup synthesis
            logger.info("[3/4] Warmup synthesis...")
            out1 = speecht5.synthesize("Testing warmup audio.", warmup_wav)
            audio1, sr1 = sf.read(out1)
            assert len(audio1) > 0, "Empty warmup audio"
            logger.info(f"Warmup audio: {len(audio1)/sr1:.2f}s at {sr1}Hz")

            # Inference synthesis
            logger.info("[4/4] Inference synthesis...")
            out2 = speecht5.synthesize("Testing inference on shared N300 device.", infer_wav)
            audio2, sr2 = sf.read(out2)
            assert len(audio2) > 0, "Empty inference audio"
            logger.info(f"Inference audio: {len(audio2)/sr2:.2f}s at {sr2}Hz")

        logger.info("=" * 60)
        logger.info("PASS: SpeechT5 standalone test")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: SpeechT5 standalone test: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    success = test_speecht5_standalone()
    sys.exit(0 if success else 1)
