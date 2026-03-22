# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test RAG system with tech_reports documents.

Uses CPU-based BGE embeddings (sentence-transformers) for compatibility.

Usage:
    cd /home/ubuntu/agentic/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/agentic/tests/test_rag_tech_reports.py
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parents[5]))

from loguru import logger


def main():
    logger.info("=" * 60)
    logger.info("RAG Tech Reports Test (CPU mode)")
    logger.info("=" * 60)

    # Step 1: Load BGE embeddings (CPU)
    logger.info("\n[1/3] Loading BGE embeddings (CPU)...")
    from models.demos.minimax_m2.agentic.tool_wrappers.bge_tool import BGETool

    start = time.time()
    bge = BGETool(mesh_device=None)  # CPU mode
    logger.info(f"BGE loaded in {time.time() - start:.1f}s")

    # Step 2: Initialize RAG
    logger.info("\n[2/3] Initializing RAG system...")
    from models.demos.minimax_m2.agentic.tool_wrappers.rag_tool import RAGTool

    rag = RAGTool(bge_tool=bge)
    logger.info("RAG initialized")

    # Step 3: Load tech_reports
    logger.info("\n[3/3] Loading tech_reports documents...")
    tech_reports_dir = Path("/home/ubuntu/agentic/tt-metal/tech_reports")

    md_files = list(tech_reports_dir.rglob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files")

    total_chunks = 0
    start = time.time()
    for i, md_file in enumerate(md_files):
        chunks = rag.add_file(str(md_file))
        total_chunks += chunks
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(md_files)} files...")

    elapsed = time.time() - start
    logger.info(f"\nIndexed {total_chunks} chunks from {len(md_files)} files in {elapsed:.1f}s")

    # Step 4: Test searches
    logger.info("\n" + "=" * 60)
    logger.info("Testing semantic search...")
    logger.info("=" * 60)

    test_queries = [
        "How do I use Flash Attention?",
        "What is the memory allocator?",
        "How to program multiple meshes?",
        "What data formats are supported?",
        "How to optimize CNN performance?",
    ]

    for query in test_queries:
        logger.info(f"\nQ: {query}")
        results = rag.search(query, top_k=2)
        for r in results:
            logger.info(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:80]}...")

    # Print stats
    logger.info("\n" + "=" * 60)
    stats = rag.stats()
    logger.info(f"RAG Stats: {stats['total_chunks']} chunks from {len(stats['sources'])} sources")
    logger.info("=" * 60)

    # Cleanup
    bge.close()
    rag.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
