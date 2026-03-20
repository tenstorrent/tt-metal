#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 0 Test: LLM standalone with shared device parameters.

Verifies LLM (Llama 3.2 3B) can:
1. Load on N300 mesh with shared device params
2. Run warmup inference
3. Run subsequent inference

NOTE: Requires HuggingFace authentication for meta-llama/Llama-3.2-3B-Instruct.
Skip this test if HF auth is unavailable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def test_llm_standalone():
    """Test LLM in isolation with shared device params."""
    logger.info("=" * 60)
    logger.info("Level 0: LLM Standalone Test")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        # Load
        logger.info("[1/3] Loading LLM (Llama 3.2 3B)...")
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

        llm = LLMTool(mesh_device=mesh)
        logger.info("LLM loaded OK")

        # Warmup
        logger.info("[2/3] Warmup inference...")
        messages = [{"role": "user", "content": "Respond with exactly: WARMUP_OK"}]
        result = llm.generate_response(messages=messages, max_new_tokens=16)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Empty response"
        logger.info(f"Warmup result: {result!r}")

        # Inference
        logger.info("[3/3] Second inference...")
        messages2 = [{"role": "user", "content": "What is 3 + 4? Answer with one token."}]
        result2 = llm.generate_response(messages=messages2, max_new_tokens=16)
        assert isinstance(result2, str), f"Expected str, got {type(result2)}"
        logger.info(f"Inference result: {result2!r}")

        logger.info("=" * 60)
        logger.info("PASS: LLM standalone test")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: LLM standalone test: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    success = test_llm_standalone()
    sys.exit(0 if success else 1)
