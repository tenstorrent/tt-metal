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
        device: ttnn.Device = None,
        model_location_generator=None,
        max_batch_size: int = 8,
        max_seq_len: int = 384,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name: str = "BAAI/bge-large-en-v1.5",
        vllm_config=None,
        prefix: str = "",
        **kwargs,
    ):
        """
        Initialize the BGE embedding model for vLLM.

        Args:
            device: TTNN device instance (required if not using vllm_config)
            model_location_generator: Optional function to generate model weight paths
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length (default 384 for BGE)
            act_dtype: Activation data type
            weight_dtype: Weight data type
            model_name: HuggingFace model name
            vllm_config: vLLM configuration (passed by vLLM wrapper)
            prefix: Model prefix (passed by vLLM wrapper)
            **kwargs: Additional arguments passed by vLLM wrapper
        """
        # Extract device from vllm_config if provided (vLLM wrapper case)
        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device

        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.model_location_generator = model_location_generator
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_name = model_name

        # Store vllm_config if provided (for vLLM wrapper compatibility)
        if vllm_config is not None:
            self.vllm_config = vllm_config

        # Load config
        self.config = transformers.BertConfig.from_pretrained(model_name)

        # Initialize runner (will be set up during first forward pass)
        self.runner: Optional[BGEPerformantRunner] = None
        self._is_initialized = False

        # Set pooler to None to satisfy vLLM wrapper (pooling is handled in forward method)
        # The wrapper will check for this and skip initialization if it exists
        self.pooler = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: transformers.PretrainedConfig,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = None,
        model_location_generator=None,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
        vllm_config=None,
        **kwargs,
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
            vllm_config: vLLM configuration (required when class is wrapped by vLLM)

        Returns:
            Initialized BGEForEmbedding instance
        """
        if max_seq_len is None:
            max_seq_len = 384  # Default for BGE-large-en-v1.5

        logger.info(
            f"Initializing BGE-Large-EN-v1.5 for vLLM: max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
        )

        # When vLLM wraps the class, it requires vllm_config to be passed
        if vllm_config is not None:
            return cls(
                device=mesh_device,
                model_location_generator=model_location_generator,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                vllm_config=vllm_config,
            )
        else:
            # Fallback for direct instantiation (not wrapped)
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

        # Match performance test: device_batch_size=8 (per device)
        # The runner infra will multiply by num_devices internally if input_ids is None
        # But we provide input_ids, so we need total batch size = 8 * num_devices
        num_devices = self.device.get_num_devices()
        per_device_batch_size = 8  # Per-device batch size (matches test_bge_e2e_performant.py line 54)
        total_batch_size = per_device_batch_size * num_devices  # Total batch size for mesh mapper

        logger.info(
            f"Initializing BGE runner: requested batch_size={batch_size}, seq_len={sequence_length}, "
            f"using device_batch_size={per_device_batch_size}, total_batch_size={total_batch_size} ({num_devices} devices)"
        )

        # Log device information for debugging
        if hasattr(self.device, "get_device_ids"):
            device_ids = self.device.get_device_ids()
            logger.info(f"BGE runner using {num_devices} devices: {device_ids}")
        else:
            logger.info(f"BGE runner using {num_devices} devices (device IDs not available)")

        # Create inputs with total batch size (matching demo.py: inputs = inputs * device.get_num_devices())
        # Extended mask shape should be [total_batch_size, 1, 1, seq_len]
        input_ids = torch.randint(
            low=0,
            high=self.config.vocab_size - 1,
            size=[total_batch_size, sequence_length],
            dtype=torch.int64,
        )
        attention_mask = torch.ones(total_batch_size, sequence_length)
        # custom_extended_mask creates [batch_size, 1, 1, seq_len] from [batch_size, seq_len]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        # Verify shape: should be [total_batch_size, 1, 1, seq_len]
        assert extended_mask.shape == (
            total_batch_size,
            1,
            1,
            sequence_length,
        ), f"Extended mask shape {extended_mask.shape} != ({total_batch_size}, 1, 1, {sequence_length})"

        token_type_ids = torch.zeros([total_batch_size, sequence_length], dtype=torch.int64)
        # Position IDs: use shape [1, seq_len] like demo.py line 45
        position_ids = torch.arange(0, sequence_length, dtype=torch.int64).unsqueeze(dim=0)

        # Initialize runner matching performance test pattern
        # Pass total_batch_size as device_batch_size since we're providing inputs with total batch size
        # The runner infra will use input_ids.shape[0] as the actual batch_size
        self.runner = BGEPerformantRunner(
            device=self.device,
            model_location_generator=self.model_location_generator,
            device_batch_size=total_batch_size,  # Pass total batch size (64) so infra tracks correct batch_size
            sequence_length=sequence_length,
            input_ids=input_ids,  # Total batch size (8 * num_devices = 64)
            extended_mask=extended_mask,  # Shape [total_batch_size, 1, 1, seq_len]
            attention_mask=attention_mask,  # Total batch size
            token_type_ids=token_type_ids,  # Total batch size
            position_ids=position_ids,  # Shape [1, seq_len] - matches demo
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
        logger.debug(f"BGE forward: processing batch_size={batch_size}, seq_len={seq_len}")

        # Ensure batch size and sequence length are within limits
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max {self.max_batch_size}"
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Initialize runner if not already done (use max_seq_len for initialization)
        self._initialize_runner(batch_size, self.max_seq_len)

        # The runner is initialized with device_batch_size=8 (per device)
        # For batches larger than 8, we need to split them into chunks
        num_devices = self.device.get_num_devices()
        per_device_batch_size = 8  # Runner initialized with device_batch_size=8
        total_batch_size = per_device_batch_size * num_devices  # Total batch size the runner can handle
        padded_seq_len = self.max_seq_len

        # Log batch processing info for debugging
        if batch_size > total_batch_size:
            logger.debug(
                f"Processing batch_size={batch_size} in chunks (total_batch_size={total_batch_size}, "
                f"num_devices={num_devices}, per_device_batch_size={per_device_batch_size})"
            )

        # The runner can handle batches up to total_batch_size (8 * num_devices)
        # For batches larger than this, we need to split them
        # However, we should avoid unnecessary splitting to minimize latency
        if batch_size > total_batch_size:
            # Split into chunks and process separately
            # Use a smaller chunk size to avoid memory issues, but process efficiently
            chunk_size = total_batch_size
            all_embeddings = []
            for i in range(0, batch_size, chunk_size):
                chunk_end = min(i + chunk_size, batch_size)
                chunk_input_ids = input_ids[i:chunk_end]
                chunk_attention_mask = attention_mask[i:chunk_end] if attention_mask is not None else None
                chunk_token_type_ids = token_type_ids[i:chunk_end] if token_type_ids is not None else None

                # Process chunk - pad to chunk_size (not total_batch_size) to avoid unnecessary padding
                chunk_embeddings = self._forward_chunk(
                    chunk_input_ids,
                    chunk_attention_mask,
                    chunk_token_type_ids,
                    position_ids,
                    chunk_size,  # Use chunk_size instead of total_batch_size for padding
                    padded_seq_len,
                )
                # Extract only the actual chunk size (remove padding)
                actual_chunk_size = chunk_end - i
                chunk_embeddings = chunk_embeddings[:actual_chunk_size]
                all_embeddings.append(chunk_embeddings)

            # Concatenate all embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            return embeddings
        else:
            # Process normally (batch size <= total_batch_size)
            # Pad to total_batch_size only if needed for mesh mapper
            return self._forward_chunk(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                total_batch_size,
                padded_seq_len,
            )

    def _forward_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        total_batch_size: int,
        padded_seq_len: int,
    ) -> torch.Tensor:
        """Process a single chunk of inputs (batch size <= total_batch_size)."""
        batch_size, seq_len = input_ids.shape

        # Pad sequence length to max_seq_len
        if input_ids.shape[1] < padded_seq_len:
            padded_input_ids = torch.zeros([batch_size, padded_seq_len], dtype=input_ids.dtype)
            padded_input_ids[:, :seq_len] = input_ids
            input_ids = padded_input_ids
        elif input_ids.shape[1] > padded_seq_len:
            input_ids = input_ids[:, :padded_seq_len]
            seq_len = padded_seq_len

        # Pad batch size to total_batch_size if needed (for mesh mapper)
        if input_ids.shape[0] < total_batch_size:
            padded_input_ids = torch.zeros([total_batch_size, input_ids.shape[1]], dtype=input_ids.dtype)
            padded_input_ids[:batch_size, :] = input_ids
            input_ids = padded_input_ids

        # Prepare inputs
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        # Pad attention mask sequence length to max_seq_len
        if attention_mask.shape[1] < padded_seq_len:
            padded_attention_mask = torch.zeros([batch_size, padded_seq_len], dtype=attention_mask.dtype)
            padded_attention_mask[:, :seq_len] = attention_mask
            attention_mask = padded_attention_mask
        elif attention_mask.shape[1] > padded_seq_len:
            attention_mask = attention_mask[:, :padded_seq_len]

        # Pad batch size to total_batch_size if needed
        if attention_mask.shape[0] < total_batch_size:
            padded_attention_mask = torch.zeros([total_batch_size, attention_mask.shape[1]], dtype=attention_mask.dtype)
            padded_attention_mask[:batch_size, :] = attention_mask
            attention_mask = padded_attention_mask

        if token_type_ids is None:
            token_type_ids = torch.zeros([batch_size, seq_len], dtype=torch.int64)

        # Pad token_type_ids sequence length to max_seq_len
        if token_type_ids.shape[1] < padded_seq_len:
            padded_token_type_ids = torch.zeros([batch_size, padded_seq_len], dtype=token_type_ids.dtype)
            padded_token_type_ids[:, :seq_len] = token_type_ids
            token_type_ids = padded_token_type_ids
        elif token_type_ids.shape[1] > padded_seq_len:
            token_type_ids = token_type_ids[:, :padded_seq_len]

        # Pad batch size to total_batch_size if needed
        if token_type_ids.shape[0] < total_batch_size:
            padded_token_type_ids = torch.zeros([total_batch_size, token_type_ids.shape[1]], dtype=token_type_ids.dtype)
            padded_token_type_ids[:batch_size, :] = token_type_ids
            token_type_ids = padded_token_type_ids

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.int64).unsqueeze(dim=0)

        # Position IDs: use shape [1, padded_seq_len] like the demo (gets broadcast automatically)
        # Pad to max sequence length
        if position_ids.shape[1] < padded_seq_len:
            padded_position_ids = torch.zeros([1, padded_seq_len], dtype=position_ids.dtype)
            padded_position_ids[0, :seq_len] = position_ids[0, :seq_len]
            padded_position_ids[0, seq_len:] = seq_len - 1  # Fill with last position ID
            position_ids = padded_position_ids
        elif position_ids.shape[1] > padded_seq_len:
            position_ids = position_ids[:, :padded_seq_len]

        # Create extended attention mask with correct shape [total_batch_size, 1, 1, padded_seq_len]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        # Verify shape matches expected: [total_batch_size, 1, 1, padded_seq_len]
        expected_shape = (total_batch_size, 1, 1, padded_seq_len)
        if extended_mask.shape != expected_shape:
            logger.warning(
                f"Extended mask shape {extended_mask.shape} != {expected_shape}. "
                f"attention_mask shape: {attention_mask.shape}. Fixing..."
            )
            # Manually create extended mask with correct shape
            extended_mask = attention_mask[:, None, None, :].to(torch.bfloat16)
            extended_mask = (1.0 - extended_mask) * torch.finfo(torch.bfloat16).min

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

        # Extract only the actual batch outputs (remove padding)
        embeddings = embeddings[:batch_size]

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

    def _init_pooler(self, vllm_config, prefix: str = ""):
        """
        Initialize pooler (required by vLLM wrapper but not used).

        Pooling is handled directly in the forward() method, so this is a no-op.
        """
        # Pooling is handled in forward() method, so no separate pooler needed
        self.pooler = None


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
