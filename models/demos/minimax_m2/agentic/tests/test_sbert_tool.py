# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test SBERTTool with our device parameters (l1_small_size=79104).
"""

import time

from loguru import logger

import ttnn


def main():
    logger.info("=" * 60)
    logger.info("SBERT Tool Test (l1_small_size=79104)")
    logger.info("=" * 60)

    # Open mesh device with our parameters
    logger.info("\n[1/4] Opening mesh device...")

    # Enable fabric for multi-chip support
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        l1_small_size=79104,
        trace_region_size=100_000_000,
        num_command_queues=2,
    )
    mesh_device.enable_program_cache()
    logger.info(f"Mesh device opened: {mesh_device.get_num_devices()} chips")

    try:
        # Create SBERT tool
        logger.info("\n[2/4] Creating SBERTTool...")
        from models.demos.minimax_m2.agentic.tool_wrappers.sbert_tool import SBERTTool

        sbert = SBERTTool(mesh_device=mesh_device)
        logger.info("SBERTTool created")

        # Test embedding generation
        logger.info("\n[3/4] Testing embedding generation...")
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning enables computers to learn from data.",
            "Tenstorrent builds AI accelerators with novel architecture.",
            "Natural language processing is a branch of AI.",
        ]

        start = time.time()
        embeddings = sbert.embed(test_texts)
        elapsed = time.time() - start

        logger.info(f"Embedded {len(test_texts)} texts in {elapsed:.3f}s")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"First embedding (first 5 dims): {embeddings[0][:5]}")

        # Test similarity search
        logger.info("\n[4/4] Testing similarity search...")
        query = "What is machine learning?"
        results = sbert.search(query, test_texts, top_k=2)

        logger.info(f"Query: '{query}'")
        for i, r in enumerate(results):
            logger.info(f"  {i+1}. [score={r['score']:.3f}] {r['document'][:50]}...")

        # Cleanup
        sbert.close()
        logger.info("\nSUCCESS!")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        logger.info("\nClosing mesh device...")
        ttnn.close_mesh_device(mesh_device)
        logger.info("Done!")


if __name__ == "__main__":
    main()
