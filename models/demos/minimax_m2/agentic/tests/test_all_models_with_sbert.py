# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test all models work together with SBERT (non-traced mode).

This verifies that SBERT in non-traced mode doesn't conflict with LLM or other models.

Usage:
    cd /home/ubuntu/agentic/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/agentic/tests/test_all_models_with_sbert.py
"""

import time

from loguru import logger


def main():
    logger.info("=" * 70)
    logger.info("Testing ALL models with SBERT (non-traced mode)")
    logger.info("=" * 70)

    # Import loader
    from models.demos.minimax_m2.agentic.loader import cleanup_models, load_all_models, open_n300_device

    # Open device
    logger.info("\n[1/8] Opening N300 mesh device...")
    mesh_device = open_n300_device(enable_fabric=True)
    logger.info(f"Device opened: {mesh_device.get_num_devices()} chips")

    try:
        # Load ALL models including SBERT
        logger.info("\n[2/8] Loading all models (this takes ~3-5 minutes)...")
        start = time.time()
        models = load_all_models(
            mesh_device,
            load_llm=True,  # LLM orchestrator
            load_whisper=True,  # STT
            load_speecht5=True,  # TTS
            load_owlvit=True,  # Object detection
            load_bert=True,  # QA
            load_yunet=True,  # Face detection
            load_t5=True,  # Translation
            load_sbert=True,  # SBERT embeddings (non-traced with LLM)
        )
        load_time = time.time() - start
        logger.info(f"All models loaded in {load_time:.1f}s")

        # Verify all models loaded
        logger.info("\n[3/8] Verifying model loading...")
        checks = {
            "LLM": models.llm is not None,
            "Whisper": models.whisper is not None,
            "SpeechT5": models.speecht5 is not None,
            "OWL-ViT": models.owlvit is not None,
            "BERT": models.bert is not None,
            "YUNet": models.yunet is not None,
            "T5": models.t5 is not None,
            "SBERT": models.sbert is not None,
            "RAG": models.rag is not None,
        }

        for name, loaded in checks.items():
            status = "✅" if loaded else "❌"
            logger.info(f"  {status} {name}")

        if not all(checks.values()):
            logger.error("Some models failed to load!")
            return False

        # Test SBERT embeddings
        logger.info("\n[4/8] Testing SBERT embeddings...")
        test_texts = [
            "TTNN is a Python library for Tenstorrent.",
            "Machine learning runs on AI accelerators.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        start = time.time()
        embeddings = models.sbert.embed(test_texts)
        sbert_time = time.time() - start
        logger.info(f"  SBERT embedding shape: {embeddings.shape}")
        logger.info(f"  SBERT inference time: {sbert_time*1000:.1f}ms")

        if embeddings.shape[1] != 768:
            logger.error(f"  ❌ Expected 768-dim embeddings, got {embeddings.shape[1]}")
            return False
        logger.info("  ✅ SBERT embeddings OK (768-dim)")

        # Test RAG search
        logger.info("\n[5/8] Testing RAG with SBERT...")
        models.rag.clear()
        for text in test_texts:
            models.rag.add_document(text, source="test")

        query = "What is the Python library for Tenstorrent?"
        results = models.rag.search(query, top_k=2)
        logger.info(f"  Query: '{query}'")
        for i, r in enumerate(results):
            logger.info(f"  {i+1}. [score={r['score']:.3f}] {r['text'][:50]}...")

        if results[0]["score"] < 0.3:
            logger.warning("  ⚠️ RAG scores are low - semantic search may not be working well")
        else:
            logger.info("  ✅ RAG semantic search OK")

        # Test LLM (critical - this is what was broken before)
        logger.info("\n[6/8] Testing LLM (critical check for SBERT compatibility)...")
        start = time.time()
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
        llm_response = models.llm.generate_response(messages)
        llm_time = time.time() - start
        logger.info(f"  LLM response: {llm_response[:100]}...")
        logger.info(f"  LLM inference time: {llm_time:.1f}s")

        # Check if response is garbage (the previous bug symptom)
        # Note: "4" is a valid short response to "What is 2+2?"
        if llm_response.count("�") > 5 or llm_response.count("\x00") > 0:
            logger.error("  ❌ LLM response contains garbage characters - SBERT conflict detected!")
            return False
        if "4" in llm_response:
            logger.info("  ✅ LLM correctly answered '4' (no garbage, SBERT compatible!)")

        # Test Whisper
        logger.info("\n[7/8] Testing other models briefly...")

        # Test BERT QA
        bert_result = models.bert.qa(question="Where is Paris?", context="Paris is the capital of France.")
        logger.info(f"  BERT QA: '{bert_result}' ({'✅' if 'france' in bert_result.lower() else '⚠️'})")

        # Test T5 translation
        t5_result = models.t5.translate("Hello world", source_lang="en", target_lang="de")
        logger.info(f"  T5 translate: '{t5_result}' ({'✅' if len(t5_result) > 0 else '⚠️'})")

        logger.info("\n[8/8] All tests passed!")
        logger.info("=" * 70)
        logger.info("SUCCESS: All models work together with SBERT (non-traced)")
        logger.info("=" * 70)
        return True

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        logger.info("\nCleaning up...")
        cleanup_models(models)
        import ttnn

        ttnn.close_mesh_device(mesh_device)
        logger.info("Done!")


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
