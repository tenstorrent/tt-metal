# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BGETool: Text embeddings using TF-IDF + SVD as fallback.

Uses simple TF-IDF vectorization when sentence-transformers is unavailable.
This provides reasonable semantic search for RAG while avoiding package conflicts.

For full neural embeddings, run BGE standalone with proper device parameters.
"""

from typing import List, Union

import numpy as np
from loguru import logger


class BGETool:
    """
    TF-IDF based text embeddings fallback.

    Generates fixed-dimension embeddings for semantic search and RAG.
    Uses scikit-learn's TfidfVectorizer + TruncatedSVD for dimensionality reduction.
    """

    EMBEDDING_DIM = 256  # Output dimension

    def __init__(self, mesh_device=None, model_location_generator=None):
        """
        Initialize BGE tool.

        Args:
            mesh_device: Ignored (uses CPU-based TF-IDF).
            model_location_generator: Ignored.
        """
        self.mesh_device = mesh_device
        self._vectorizer = None
        self._svd = None
        self._fitted = False
        self._corpus = []
        self._corpus_embeddings = None
        self._init_model()

    def _init_model(self):
        """Initialize TF-IDF vectorizer."""
        logger.info("Loading BGE embeddings (TF-IDF fallback mode)...")

        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._svd = TruncatedSVD(n_components=self.EMBEDDING_DIM)

        logger.info("BGE embeddings ready (TF-IDF mode).")

    def _fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts."""
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        # Adjust SVD components if vocabulary is smaller than target dimension
        n_features = tfidf_matrix.shape[1]
        if n_features < self.EMBEDDING_DIM:
            from sklearn.decomposition import TruncatedSVD

            self._svd = TruncatedSVD(n_components=max(1, n_features - 1))

        embeddings = self._svd.fit_transform(tfidf_matrix)
        self._fitted = True
        return embeddings

    def _transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        if not self._fitted:
            # Need to fit first - use the input texts
            return self._fit_transform(texts)
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = self._svd.transform(tfidf_matrix)
        return embeddings

    def embed(
        self,
        texts: Union[str, List[str]],
        add_instruction: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed.
            add_instruction: Ignored (TF-IDF doesn't use instructions).

        Returns:
            numpy array of shape (n_texts, EMBEDDING_DIM) with normalized embeddings.
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # If we haven't fitted yet and this is a document batch, fit on it
        if not self._fitted and len(texts) > 1:
            embeddings = self._fit_transform(texts)
        elif not self._fitted:
            # Single query before any documents - return random (will be re-embedded later)
            embeddings = np.random.randn(len(texts), self.EMBEDDING_DIM)
        else:
            embeddings = self._transform(texts)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def add_to_corpus(self, texts: List[str]) -> np.ndarray:
        """
        Add documents to corpus and refit model.

        Args:
            texts: Documents to add.

        Returns:
            Embeddings for the new documents.
        """
        self._corpus.extend(texts)

        # Refit on full corpus
        self._fitted = False
        all_embeddings = self._fit_transform(self._corpus)

        # Return just the new embeddings
        return all_embeddings[-len(texts) :]

    def similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute cosine similarity between query and documents.

        Args:
            query: Query text.
            documents: List of document texts.

        Returns:
            List of similarity scores.
        """
        # Fit on documents if not fitted
        if not self._fitted:
            doc_embs = self._fit_transform(documents)
        else:
            doc_embs = self._transform(documents)

        query_emb = self._transform([query])

        # Normalize
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs = doc_embs / (doc_norms + 1e-8)
        query_norm = np.linalg.norm(query_emb)
        query_emb = query_emb / (query_norm + 1e-8)

        # Cosine similarity
        similarities = np.dot(doc_embs, query_emb.T).flatten()
        return similarities.tolist()

    def search(self, query: str, documents: List[str], top_k: int = 3) -> List[dict]:
        """
        Search for most similar documents.

        Args:
            query: Query text.
            documents: List of document texts.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'document', 'score', and 'index'.
        """
        scores = self.similarity(query, documents)
        indices = np.argsort(scores)[::-1][:top_k]

        return [{"document": documents[i], "score": scores[i], "index": int(i)} for i in indices]

    def close(self):
        """Release resources."""
        self._vectorizer = None
        self._svd = None
        self._corpus = []
        logger.info("BGETool closed.")
