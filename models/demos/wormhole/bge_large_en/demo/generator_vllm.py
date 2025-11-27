# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Integration for BGE-Large-EN-v1.5 Embedding Model

This module provides vLLM integration for the BGE embedding model, enabling
OpenAI Embedding API compatibility through vLLM's serving infrastructure.

Usage:
    The model can be registered with vLLM's ModelRegistry and served via
    the OpenAI Embedding API endpoint.
"""

from typing import Optional

import torch
import transformers
from loguru import logger

import ttnn
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.sentence_bert.reference.sentence_bert import custom_extended_mask


class BGEForEmbedding:
    """
    vLLM-compatible wrapper for BGE-Large-EN-v1.5 embedding model.

    This class implements the interface required by vLLM for embedding models,
    enabling OpenAI Embedding API compatibility.
    """

    def __init__(
        self,
        device: ttnn.Device,
        model_location_generator=None,
        max_batch_size: int = 8,
        max_seq_len: int = 384,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name: str = "BAAI/bge-large-en-v1.5",
    ):
        """
        Initialize the BGE embedding model for vLLM.

        Args:
            device: TTNN device instance
            model_location_generator: Optional function to generate model weight paths
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length (default 384 for BGE)
            act_dtype: Activation data type
            weight_dtype: Weight data type
            model_name: HuggingFace model name
        """
        self.device = device
        self.model_location_generator = model_location_generator
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_name = model_name

        # Load config
        self.config = transformers.BertConfig.from_pretrained(model_name)

        # Initialize runner (will be set up during first forward pass)
        self.runner: Optional[BGEPerformantRunner] = None
        self._is_initialized = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: transformers.PretrainedConfig,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = None,
        model_location_generator=None,
    ) -> "BGEForEmbedding":
        """
        Initialize the model for vLLM.

        This is the entry point called by vLLM's TTModelLoader.

        Args:
            hf_config: HuggingFace model configuration
            mesh_device: TTNN mesh device
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length (defaults to 384 for BGE)
            model_location_generator: Optional function to generate model weight paths

        Returns:
            Initialized BGEForEmbedding instance
        """
        if max_seq_len is None:
            max_seq_len = 384  # Default for BGE-large-en-v1.5

        logger.info(
            f"Initializing BGE-Large-EN-v1.5 for vLLM: " f"max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
        )

        return cls(
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    def _initialize_runner(self, batch_size: int, sequence_length: int):
        """Initialize the BGE runner with the given batch size and sequence length."""
        if self._is_initialized and self.runner is not None:
            return

        logger.info(f"Initializing BGE runner: batch_size={batch_size}, seq_len={sequence_length}")

        # Create dummy inputs for initialization
        input_ids = torch.randint(
            low=0,
            high=self.config.vocab_size - 1,
            size=[batch_size, sequence_length],
            dtype=torch.int64,
        )
        attention_mask = torch.ones(batch_size, sequence_length)
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = torch.zeros([batch_size, sequence_length], dtype=torch.int64)
        position_ids = torch.arange(0, sequence_length, dtype=torch.int64).unsqueeze(dim=0)

        # Initialize runner
        self.runner = BGEPerformantRunner(
            device=self.device,
            model_location_generator=self.model_location_generator,
            device_batch_size=batch_size,
            sequence_length=sequence_length,
            input_ids=input_ids,
            extended_mask=extended_mask,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            act_dtype=self.act_dtype,
            weight_dtype=self.weight_dtype,
            model_name=self.model_name,
        )

        # Capture trace for optimized inference
        self.runner._capture_bge_trace_2cqs()
        self._is_initialized = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for embedding generation.

        This is the main interface method called by vLLM for embedding inference.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask tensor
            token_type_ids: Optional token type IDs tensor
            position_ids: Optional position IDs tensor

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Ensure batch size and sequence length are within limits
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max {self.max_batch_size}"
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Initialize runner if not already done
        self._initialize_runner(batch_size, seq_len)

        # Prepare inputs
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        if token_type_ids is None:
            token_type_ids = torch.zeros([batch_size, seq_len], dtype=torch.int64)

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.int64).unsqueeze(dim=0).repeat(batch_size, 1)

        # Create extended attention mask
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)

        # Run inference
        output = self.runner.run(
            input_ids=input_ids,
            tokens=token_type_ids,
            posids=position_ids,
            ext_att_mask=extended_mask,
            att_mask=attention_mask,
        )

        # Convert TTNN tensor to PyTorch tensor
        embeddings = ttnn.to_torch(
            output,
            dtype=torch.float32,
            mesh_composer=self.runner.runner_infra.output_mesh_composer,
        )

        # Squeeze batch dimension if needed (BGE returns shape [batch_size, 1, embedding_dim])
        if embeddings.dim() == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.config.hidden_size

    def get_max_seq_len(self) -> int:
        """Return the maximum sequence length."""
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        """Return the maximum batch size."""
        return self.max_batch_size


def register_model():
    """
    Register the BGE model with vLLM's ModelRegistry.

    This function should be called to make the model available to vLLM.
    Typically called from vLLM's model registration code.
    """
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        ModelRegistry.register_model(
            "BAAI/bge-large-en-v1.5",
            BGEForEmbedding,
            architecture="BertModel",
        )
        logger.info("Successfully registered BGE-Large-EN-v1.5 with vLLM ModelRegistry")
    except ImportError:
        logger.warning(
            "vLLM ModelRegistry not available. "
            "Make sure vLLM is installed and the model is registered in vLLM's model loader."
        )
