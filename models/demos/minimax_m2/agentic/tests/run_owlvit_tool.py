#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone test: OWLViTTool on N300."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from loguru import logger
from PIL import Image

import ttnn


def make_test_image(path):
    data = np.zeros((512, 512, 3), dtype=np.uint8)
    data[40:200, 40:200] = [220, 30, 30]  # red block
    data[40:200, 312:472] = [30, 30, 220]  # blue block
    data[312:472, 176:336] = [30, 180, 30]  # green block
    Image.fromarray(data, "RGB").save(path)
    return path


def main():
    logger.info("Opening single device for OWL-ViT...")
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = make_test_image(f"{tmpdir}/test.png")

        try:
            from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

            logger.info("Loading OWLViTTool...")
            owl = OWLViTTool(mesh_device=device)

            # First call - triggers compilation
            logger.info("First call (compilation)...")
            results = owl.detect(img_path, "red block, blue block, green block")
            logger.info(f"Detections: {len(results)} objects")
            for d in results:
                logger.info(f"  {d['label']} score={d['score']:.3f}")

            # Second call - trace reuse
            logger.info("Second call (trace reuse)...")
            results2 = owl.detect(img_path, "coloured square")
            logger.info(f"Detections (2nd): {len(results2)} objects")

            # Format check
            for d in results:
                assert "label" in d and "score" in d and "bbox" in d, "Bad detection format"
                assert 0.0 <= d["score"] <= 1.0, f"Bad score: {d['score']}"
                assert len(d["bbox"]) == 4, "Bad bbox"

            logger.info("OWL-ViT tool: ALL DONE")
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
