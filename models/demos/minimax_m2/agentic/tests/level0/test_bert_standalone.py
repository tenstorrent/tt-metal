#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 0 Test: BERT standalone with shared device parameters.

Verifies BERT can:
1. Load on N300 mesh with shared device params
2. Run warmup inference
3. Run subsequent inference

Key hypothesis: Does BERT work standalone with l1_small_size=24576?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def test_bert_standalone():
    """Test BERT in isolation with shared device params."""
    logger.info("=" * 60)
    logger.info("Level 0: BERT Standalone Test")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        # Load
        logger.info("[1/3] Loading BERT...")
        from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

        bert = BERTTool(mesh_device=mesh)
        logger.info("BERT loaded OK")

        # Warmup
        logger.info("[2/3] Warmup inference...")
        question = "How many chips does the N300 have?"
        context = "The N300 contains two Wormhole B0 chips."
        result = bert.qa(question, context)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result.strip()) > 0, "Empty answer"
        logger.info(f"Warmup result: {result!r}")

        # Inference
        logger.info("[3/3] Second inference...")
        question2 = "What type of chips are in N300?"
        context2 = "The N300 uses two Wormhole B0 chips connected by high-bandwidth Ethernet."
        result2 = bert.qa(question2, context2)
        assert isinstance(result2, str), f"Expected str, got {type(result2)}"
        logger.info(f"Inference result: {result2!r}")

        logger.info("=" * 60)
        logger.info("PASS: BERT standalone test")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: BERT standalone test: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    success = test_bert_standalone()
    sys.exit(0 if success else 1)
