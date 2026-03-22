# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test RAG pipeline with SBERT on TTNN.
"""

import time
from pathlib import Path

from loguru import logger

import ttnn


def main():
    logger.info("=" * 60)
    logger.info("RAG + SBERT Test (TTNN accelerated embeddings)")
    logger.info("=" * 60)

    # Open mesh device with our parameters
    logger.info("\n[1/5] Opening mesh device...")

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
        logger.info("\n[2/5] Creating SBERTTool...")
        from models.demos.minimax_m2.agentic.tool_wrappers.sbert_tool import SBERTTool

        sbert = SBERTTool(mesh_device=mesh_device)
        logger.info("SBERTTool created")

        # Create RAG tool
        logger.info("\n[3/5] Creating RAGTool...")
        from models.demos.minimax_m2.agentic.tool_wrappers.rag_tool import RAGTool

        rag = RAGTool(bge_tool=sbert)  # RAG uses same interface
        logger.info("RAGTool created")

        # Ingest tech_reports documents
        logger.info("\n[4/5] Ingesting tech_reports documents...")
        tech_reports_dir = Path("/home/ubuntu/agentic/tt-metal/tech_reports")

        if not tech_reports_dir.exists():
            logger.warning(f"tech_reports directory not found: {tech_reports_dir}")
            # Use some test documents instead
            test_docs = [
                "Tenstorrent develops AI accelerators using a novel architecture called Tensix.",
                "The Wormhole chip contains multiple Tensix cores for parallel matrix operations.",
                "TTNN is a Python library for deploying neural networks on Tenstorrent hardware.",
                "Sentence BERT generates dense embeddings for semantic similarity tasks.",
                "RAG combines retrieval with generation for knowledge-grounded responses.",
            ]
            for i, doc in enumerate(test_docs):
                rag.add_document(doc, source=f"test_doc_{i}")
        else:
            start = time.time()
            # Add up to 10 markdown files for quick testing
            md_files = list(tech_reports_dir.rglob("*.md"))[:10]
            for f in md_files:
                try:
                    rag.add_file(str(f))
                except Exception as e:
                    logger.warning(f"Failed to add {f.name}: {e}")
            elapsed = time.time() - start
            logger.info(f"Ingested {len(md_files)} files in {elapsed:.1f}s")

        # Show stats
        stats = rag.stats()
        logger.info(f"RAG stats: {stats['total_chunks']} chunks, embedding_dim={stats['embedding_dim']}")

        # Test retrieval
        logger.info("\n[5/5] Testing retrieval...")
        test_queries = [
            "What is TTNN?",
            "How does the Tensix core work?",
            "What are the memory constraints?",
        ]

        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            start = time.time()
            results = rag.search(query, top_k=2)
            elapsed = time.time() - start
            logger.info(f"Search took {elapsed*1000:.1f}ms")
            for i, r in enumerate(results):
                preview = r["text"][:80].replace("\n", " ")
                logger.info(f"  {i+1}. [score={r['score']:.3f}] [{r['source']}] {preview}...")

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
