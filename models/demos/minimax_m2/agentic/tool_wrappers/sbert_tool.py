# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SBERTTool: Sentence BERT embeddings on TTNN for RAG.

Uses sentence-transformers/all-MiniLM-L6-v2 compatible BERT model
running on Tenstorrent hardware for fast sentence embeddings.
"""

from typing import List, Union

import numpy as np
import torch
import transformers
from loguru import logger

import ttnn
from models.demos.sentence_bert.reference.sentence_bert import custom_extended_mask
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner


class SBERTTool:
    """
    TTNN-accelerated Sentence BERT for text embeddings.

    Generates 384-dimensional embeddings for semantic search and RAG.
    Runs on Tenstorrent N300 at ~780 sentences/sec (2 chips) with trace.

    Args:
        mesh_device: TTNN mesh device
        model_location_generator: Optional model path generator
        use_trace: If True, capture trace for fast execution. If False, run without
                   trace (slower but compatible with LLM multi-model setup).
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    SEQ_LENGTH = 384
    BATCH_SIZE = 8
    EMBEDDING_DIM = 768  # BERT base hidden dimension

    def __init__(self, mesh_device, model_location_generator=None, use_trace: bool = True):
        self.mesh_device = mesh_device
        self.model_location_generator = model_location_generator
        self._runner = None
        self._tokenizer = None
        self._use_trace = use_trace
        self._num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        self._init_model()

    def _get_model_location_generator(self):
        """Get or create model location generator."""
        if self.model_location_generator is not None:
            return self.model_location_generator

        def model_location_generator(model_version, model_subdir=""):
            from models.common.utility_functions import get_model_prefix

            model_prefix = get_model_prefix()
            return f"{model_prefix}/sentence_bert/{model_version}"

        return model_location_generator

    def _init_model(self):
        """Initialize Sentence BERT model components."""
        logger.info(f"Loading Sentence BERT embeddings...")

        # Load tokenizer
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.MODEL_NAME)

        # Device batch size is multiplied by num_devices internally
        device_batch = self.BATCH_SIZE * self._num_devices

        # Create dummy inputs for runner initialization
        dummy_input_ids = torch.randint(low=0, high=30000, size=[device_batch, self.SEQ_LENGTH], dtype=torch.int64)
        dummy_attention_mask = torch.ones(device_batch, self.SEQ_LENGTH)
        dummy_extended_mask = custom_extended_mask(dummy_attention_mask, dtype=torch.bfloat16)
        dummy_token_type_ids = torch.zeros([device_batch, self.SEQ_LENGTH], dtype=torch.int64)
        dummy_position_ids = torch.arange(0, self.SEQ_LENGTH, dtype=torch.int64).unsqueeze(dim=0)

        # Initialize runner with dummy inputs
        self._runner = SentenceBERTPerformantRunner(
            device=self.mesh_device,
            device_batch_size=self.BATCH_SIZE,
            sequence_length=self.SEQ_LENGTH,
            model_location_generator=self._get_model_location_generator(),
            input_ids=dummy_input_ids,
            extended_mask=dummy_extended_mask,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
            position_ids=dummy_position_ids,
        )

        if self._use_trace:
            # Capture trace for fast execution (conflicts with LLM multi-model setup)
            logger.info("Capturing Sentence BERT trace...")
            self._runner._capture_sentencebert_trace_2cqs()
        else:
            # Non-traced mode: slower but compatible with LLM
            logger.info("Sentence BERT in non-traced mode (LLM compatible)...")
            # Setup inputs and run once to warm up JIT compilation
            (
                ttnn_input_ids,
                input_mem_config,
                ttnn_token_ids,
                ttnn_pos_ids,
                ttnn_ext_att_mask,
                ttnn_att_mask,
            ) = self._runner.runner_infra.setup_l1_sharded_input()

            # Move to device with sharded memory config
            self._runner.runner_infra.ttnn_input_ids = ttnn.to_memory_config(
                ttnn_input_ids.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_token_ids = ttnn.to_memory_config(
                ttnn_token_ids.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(
                ttnn_pos_ids.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(
                ttnn_ext_att_mask.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_att_mask = ttnn.to_memory_config(
                ttnn_att_mask.to(self.mesh_device), input_mem_config
            )

            # Store input_mem_config for later use
            self._input_mem_config = input_mem_config

            # Run once to warm up JIT compilation
            self._runner.runner_infra.run()
            self._runner.runner_infra.dealloc_output()

        logger.info("Sentence BERT embeddings ready.")

    def _prepare_inputs(self, texts: List[str]):
        """Tokenize and prepare inputs for the model."""
        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].float()

        # Create extended mask for attention
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)

        # Token type IDs (zeros for single sentences)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)

        # Position IDs
        position_ids = torch.arange(0, self.SEQ_LENGTH, dtype=torch.int64).unsqueeze(dim=0)

        return input_ids, token_type_ids, position_ids, extended_mask, attention_mask

    def embed(
        self,
        texts: Union[str, List[str]],
        add_instruction: bool = True,  # Ignored for compatibility
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed.
            add_instruction: Ignored (for API compatibility with BGE).

        Returns:
            numpy array of shape (n_texts, 384) with normalized embeddings.
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        original_len = len(texts)
        effective_batch = self.BATCH_SIZE * self._num_devices

        # Pad to batch size
        if original_len < effective_batch:
            texts = texts + [""] * (effective_batch - original_len)
        elif original_len > effective_batch:
            # Process in batches
            all_embeddings = []
            for i in range(0, original_len, effective_batch):
                batch = texts[i : i + effective_batch]
                batch_emb = self.embed(batch)
                all_embeddings.append(batch_emb)
            return np.vstack(all_embeddings)[:original_len]

        # Prepare inputs
        input_ids, token_type_ids, position_ids, extended_mask, attention_mask = self._prepare_inputs(texts)

        # Run model (traced or non-traced)
        if self._use_trace:
            ttnn_out = self._runner.run(
                input_ids=input_ids,
                tokens=token_type_ids,
                posids=position_ids,
                ext_att_mask=extended_mask,
                att_mask=attention_mask,
            )
        else:
            # Non-traced execution: setup inputs and run directly
            (
                ttnn_input_ids,
                input_mem_config,
                ttnn_token_ids,
                ttnn_pos_ids,
                ttnn_ext_att_mask,
                ttnn_att_mask,
            ) = self._runner.runner_infra.setup_l1_sharded_input(
                input_ids, token_type_ids, position_ids, extended_mask, attention_mask
            )

            # Move to device with sharded memory config
            self._runner.runner_infra.ttnn_input_ids = ttnn.to_memory_config(
                ttnn_input_ids.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_token_ids = ttnn.to_memory_config(
                ttnn_token_ids.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(
                ttnn_pos_ids.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(
                ttnn_ext_att_mask.to(self.mesh_device), input_mem_config
            )
            self._runner.runner_infra.ttnn_att_mask = ttnn.to_memory_config(
                ttnn_att_mask.to(self.mesh_device), input_mem_config
            )

            self._runner.runner_infra.run()
            ttnn_out = self._runner.runner_infra.ttnn_output_tensor[0]
            # Note: Don't dealloc_output() here - we need the tensor for to_torch() below

        # Convert output - sentence BERT outputs last hidden state [batch, seq, hidden]
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if self._num_devices > 1 else None

        embeddings = ttnn.to_torch(ttnn_out, dtype=torch.float32, mesh_composer=mesh_composer)

        # Deallocate output tensor after conversion (for non-traced mode)
        if not self._use_trace:
            self._runner.runner_infra.dealloc_output()

        # Mean pooling over sequence dimension (using attention mask)
        # embeddings shape: [batch, seq_len, hidden_dim]
        if len(embeddings.shape) == 3:
            # Apply attention mask for mean pooling
            attn_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            masked_embeddings = embeddings * attn_mask
            sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = attn_mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
            embeddings = sum_embeddings / sum_mask
        else:
            # Already pooled
            embeddings = embeddings.squeeze()

        embeddings = embeddings.cpu().numpy()

        # Ensure 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings[:original_len]

    def similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute cosine similarity between query and documents.

        Args:
            query: Query text.
            documents: List of document texts.

        Returns:
            List of similarity scores (0-1).
        """
        query_emb = self.embed(query)
        doc_embs = self.embed(documents)

        # Cosine similarity (embeddings are already normalized)
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
        if self._runner is not None:
            try:
                if self._use_trace:
                    self._runner.release()  # Release trace
            except Exception as e:
                logger.warning(f"SBERT runner release failed: {e}")
            self._runner = None
        logger.info("SBERTTool closed.")
