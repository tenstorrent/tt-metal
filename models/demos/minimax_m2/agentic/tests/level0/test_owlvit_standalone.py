#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 0 Test: OWL-ViT standalone with shared device parameters.

Verifies OWL-ViT can:
1. Load on chip0 submesh with shared device params
2. Run warmup inference
3. Run subsequent inference
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def make_test_image(path: str):
    """Create a simple test image with colored blocks."""
    data = np.zeros((512, 512, 3), dtype=np.uint8)
    data[40:200, 40:200] = [220, 30, 30]  # red block
    data[312:472, 176:336] = [30, 180, 30]  # green block
    Image.fromarray(data, "RGB").save(path)


def test_owlvit_standalone():
    """Test OWL-ViT in isolation with shared device params."""
    logger.info("=" * 60)
    logger.info("Level 0: OWL-ViT Standalone Test")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        # Create chip0 submesh (OWL-ViT needs single-buffer tensors)
        chip0 = (
            mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)) if mesh.get_num_devices() > 1 else mesh
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/test.png"
            make_test_image(img_path)

            # Load
            logger.info("[1/3] Loading OWL-ViT...")
            from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

            owlvit = OWLViTTool(mesh_device=chip0)
            logger.info("OWL-ViT loaded OK")

            # Warmup
            logger.info("[2/3] Warmup inference...")
            detections = owlvit.detect(img_path, "red block, green block")
            assert isinstance(detections, list), f"Expected list, got {type(detections)}"
            logger.info(f"Warmup found {len(detections)} detections")

            # Inference
            logger.info("[3/3] Second inference...")
            detections2 = owlvit.detect(img_path, "colored shape")
            assert isinstance(detections2, list), f"Expected list, got {type(detections2)}"
            logger.info(f"Inference found {len(detections2)} detections")

        logger.info("=" * 60)
        logger.info("PASS: OWL-ViT standalone test")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: OWL-ViT standalone test: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    success = test_owlvit_standalone()
    sys.exit(0 if success else 1)
