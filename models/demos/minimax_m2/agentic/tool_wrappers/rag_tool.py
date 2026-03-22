# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RAGTool: Retrieval-Augmented Generation using embeddings + cosine similarity.

Provides document ingestion, embedding generation, and semantic retrieval
for enhancing LLM responses with relevant context.

Uses TF-IDF based embeddings in fallback mode (when neural embeddings unavailable).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class RAGTool:
    """
    RAG system using embeddings + cosine similarity search.

    Supports:
    - Document ingestion from text files or strings
    - Automatic chunking of long documents
    - Semantic search with configurable top-k
    - Persistent index save/load
    """

    CHUNK_SIZE = 512  # tokens (roughly 2000 chars)
    CHUNK_OVERLAP = 50  # tokens overlap between chunks

    def __init__(self, bge_tool, index_path: Optional[str] = None):
        """
        Initialize RAG tool.

        Args:
            bge_tool: Initialized BGETool for embeddings.
            index_path: Optional path to load/save index.
        """
        self.bge = bge_tool
        self.index_path = index_path

        # Document storage
        self.documents: List[str] = []  # Original chunks
        self.metadata: List[Dict[str, Any]] = []  # Chunk metadata (source, etc.)
        self.embeddings: Optional[np.ndarray] = None  # Cached embeddings

        # Load existing index if provided
        if index_path and os.path.exists(f"{index_path}.json"):
            self.load(index_path)

        logger.info(f"RAGTool initialized. Documents: {len(self.documents)}")

    def _chunk_text(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        # Simple character-based chunking (roughly 4 chars per token)
        char_chunk_size = self.CHUNK_SIZE * 4
        char_overlap = self.CHUNK_OVERLAP * 4

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + char_chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                if last_period > char_chunk_size // 2:
                    chunk_text = chunk_text[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_char": start,
                }
            )

            start = end - char_overlap
            chunk_id += 1

        return chunks

    def _recompute_embeddings(self):
        """Recompute embeddings for all documents."""
        if not self.documents:
            self.embeddings = None
            return

        # Generate embeddings for all documents
        self.embeddings = self.bge.embed(self.documents, add_instruction=False)

        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

    def add_document(self, text: str, source: str = "document") -> int:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text.
            source: Source identifier (filename, URL, etc.).

        Returns:
            Number of chunks added.
        """
        chunks = self._chunk_text(text, source)

        if not chunks:
            return 0

        # Store documents and metadata
        for chunk in chunks:
            self.documents.append(chunk["text"])
            self.metadata.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "start_char": chunk["start_char"],
                }
            )

        # Recompute all embeddings (needed for TF-IDF mode)
        self._recompute_embeddings()

        logger.info(f"Added {len(chunks)} chunks from '{source}'. Total: {len(self.documents)}")
        return len(chunks)

    def add_file(self, file_path: str) -> int:
        """
        Add a text file to the knowledge base.

        Args:
            file_path: Path to text file.

        Returns:
            Number of chunks added.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return 0

        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.add_document(text, source=path.name)

    def add_directory(self, dir_path: str, extensions: List[str] = None) -> int:
        """
        Add all text files from a directory.

        Args:
            dir_path: Path to directory.
            extensions: File extensions to include (default: .txt, .md, .py)

        Returns:
            Total number of chunks added.
        """
        if extensions is None:
            extensions = [".txt", ".md", ".py", ".json", ".yaml", ".yml"]

        path = Path(dir_path)
        if not path.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            return 0

        # Collect all files first to batch the embedding
        all_chunks = []
        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = self._chunk_text(text, file_path.name)
                all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Store all documents
        for chunk in all_chunks:
            self.documents.append(chunk["text"])
            self.metadata.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "start_char": chunk["start_char"],
                }
            )

        # Compute embeddings once for all
        self._recompute_embeddings()

        logger.info(f"Added {len(all_chunks)} chunks from directory. Total: {len(self.documents)}")
        return len(all_chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'text', 'score', 'source', 'index'.
        """
        if len(self.documents) == 0 or self.embeddings is None:
            return []

        # Generate query embedding
        query_emb = self.bge.embed(query, add_instruction=True)

        # Normalize
        query_emb = query_emb.flatten()
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_emb)

        # Get top-k indices
        k = min(top_k, len(self.documents))
        indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in indices:
            results.append(
                {
                    "text": self.documents[idx],
                    "score": float(scores[idx]),
                    "source": self.metadata[idx]["source"],
                    "index": int(idx),
                }
            )

        return results

    def get_context(self, query: str, top_k: int = 3, max_chars: int = 4000) -> str:
        """
        Get formatted context for LLM.

        Args:
            query: Search query.
            top_k: Number of chunks to retrieve.
            max_chars: Maximum context length.

        Returns:
            Formatted context string.
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return "No relevant documents found in the knowledge base."

        context_parts = []
        total_chars = 0

        for i, r in enumerate(results, 1):
            chunk = f"[Source: {r['source']}]\n{r['text']}"
            if total_chars + len(chunk) > max_chars:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)

        return "\n\n---\n\n".join(context_parts)

    def query_with_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Search and return structured result for LLM tool use.

        Args:
            query: User's question.
            top_k: Number of documents to retrieve.

        Returns:
            Dict with 'context', 'sources', and 'num_results'.
        """
        results = self.search(query, top_k=top_k)
        context = self.get_context(query, top_k=top_k)

        sources = list(set(r["source"] for r in results))

        return {
            "context": context,
            "sources": sources,
            "num_results": len(results),
            "query": query,
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save index and documents to disk."""
        save_path = path or self.index_path
        if not save_path:
            logger.warning("No save path specified")
            return

        # Save documents, metadata, and embeddings
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
        }
        with open(f"{save_path}.json", "w") as f:
            json.dump(data, f)

        logger.info(f"Saved RAG index to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load index and documents from disk."""
        load_path = path or self.index_path
        if not load_path:
            logger.warning("No load path specified")
            return

        json_path = f"{load_path}.json"

        if not os.path.exists(json_path):
            logger.warning(f"Index file not found at {load_path}")
            return

        # Load documents, metadata, and embeddings
        with open(json_path, "r") as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.metadata = data["metadata"]
        if data.get("embeddings"):
            self.embeddings = np.array(data["embeddings"])
        else:
            self._recompute_embeddings()

        logger.info(f"Loaded RAG index from {load_path}. Documents: {len(self.documents)}")

    def clear(self) -> None:
        """Clear all documents and reset index."""
        self.documents = []
        self.metadata = []
        self.embeddings = None
        logger.info("RAG index cleared")

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        sources = {}
        for m in self.metadata:
            src = m["source"]
            sources[src] = sources.get(src, 0) + 1

        return {
            "total_chunks": len(self.documents),
            "sources": sources,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
        }

    def close(self):
        """Release resources."""
        if self.index_path:
            self.save()
        logger.info("RAGTool closed.")
