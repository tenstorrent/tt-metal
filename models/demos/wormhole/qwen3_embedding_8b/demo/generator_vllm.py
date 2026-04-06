# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Integration for Qwen3-Embedding-8B Model

This module provides vLLM integration for the Qwen3-Embedding model, enabling
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
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.generator_vllm import initialize_vllm_text_transformer
from models.tt_transformers.tt.model_config import DecodersPrecision


class Qwen3ForEmbedding:
    """
    vLLM-compatible wrapper for Qwen3-Embedding-8B embedding model.

    This class implements the interface required by vLLM for embedding models,
    enabling OpenAI Embedding API compatibility.
    """

    def __init__(
        self,
        device: ttnn.Device = None,
        model_location_generator=None,
        max_batch_size: int = 8,
        max_seq_len: int = 8192,  # Qwen3-Embedding supports up to 8192
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        vllm_config=None,
        prefix: str = "",
        **kwargs,
    ):
        """
        Initialize the Qwen3-Embedding model for vLLM.

        Args:
            device: TTNN device instance (required if not using vllm_config)
            model_location_generator: Optional function to generate model weight paths
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length (default 8192 for Qwen3-Embedding)
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
        self.config = transformers.AutoConfig.from_pretrained(model_name)

        # Initialize model and generator (will be set up during first forward pass)
        self.model = None
        self.model_args = None
        self.generator = None
        self.processor = None
        self.tokenizer = None
        self._is_initialized = False
        self._kv_cache = None  # Will be allocated during initialization
        self.paged_attention_config = None  # Will be set during initialization

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
    ) -> "Qwen3ForEmbedding":
        """
        Initialize the model for vLLM.

        This is the entry point called by vLLM's TTModelLoader.

        Args:
            hf_config: HuggingFace model configuration
            mesh_device: TTNN mesh device
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length (defaults to 8192 for Qwen3-Embedding)
            model_location_generator: Optional function to generate model weight paths
            vllm_config: vLLM configuration (required when class is wrapped by vLLM)

        Returns:
            Initialized Qwen3ForEmbedding instance
        """
        if max_seq_len is None:
            max_seq_len = 8192  # Default for Qwen3-Embedding-8B

        logger.info(
            f"Initializing Qwen3-Embedding-8B for vLLM: " f"max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
        )

        # When vLLM wraps the class, it requires vllm_config to be passed
        if vllm_config is not None:
            # Mark this as an embedding model in override_tt_config for KV cache allocation
            # This flag is used by get_num_available_blocks_tt to allocate sufficient blocks
            if (
                not hasattr(vllm_config.model_config, "override_tt_config")
                or vllm_config.model_config.override_tt_config is None
            ):
                vllm_config.model_config.override_tt_config = {}
            vllm_config.model_config.override_tt_config["is_embedding_model"] = True

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

    def _initialize_model(self, batch_size: int, sequence_length: int):
        """Initialize the Qwen3-Embedding model with the given batch size and sequence length."""
        if self._is_initialized and self.model is not None:
            return

        logger.info(
            f"Initializing Qwen3-Embedding model: requested batch_size={batch_size}, "
            f"seq_len={sequence_length}, max_seq_len={self.max_seq_len}"
        )

        # Use tt_transformers infrastructure to initialize the model
        # Similar to how generator_vllm.py initializes text transformers
        dtype = self.weight_dtype

        # Initialize the transformer model using tt_transformers infrastructure
        # Note: DecodersPrecision.performance takes (num_decoders, model_name)
        # initialize_vllm_text_transformer wraps it and calls with (n_layers, model_name)
        # So we can pass both arguments directly
        def optimizations_wrapper(n_layers, model_name):
            # Call DecodersPrecision.performance with both num_decoders and model_name
            return DecodersPrecision.performance(n_layers, model_name)

        # Create paged attention config for paged KV cache
        # Calculate max_num_blocks based on max_seq_len
        # block_size is typically 32, so max_num_blocks = max_seq_len / block_size per batch
        # For embedding models, we need enough blocks for max_seq_len * max_batch_size
        block_size = 32  # Standard block size for paged attention
        # Calculate max_num_blocks: need enough for max_batch_size sequences of max_seq_len
        # Each sequence needs ceil(max_seq_len / block_size) blocks
        blocks_per_seq = (self.max_seq_len + block_size - 1) // block_size
        max_num_blocks = blocks_per_seq * self.max_batch_size
        # Ensure it's at least 1024 (common minimum)
        max_num_blocks = max(max_num_blocks, 1024)

        self.paged_attention_config = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

        logger.info(
            f"Using paged attention: block_size={block_size}, max_num_blocks={max_num_blocks}, "
            f"blocks_per_seq={blocks_per_seq}"
        )

        self.model, self.model_args = initialize_vllm_text_transformer(
            hf_config=self.config,
            tt_data_parallel=1,  # Use data parallel from device configuration
            mesh_device=self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dtype=dtype,
            optimizations=optimizations_wrapper,
        )

        # Set paged_attention_config on all attention layers
        # The model was created with use_paged_kv_cache=True, but paged_attention_config was None
        # We need to set it and reinitialize the KV cache with the correct paged attention shape
        for model_idx, model in enumerate(self.model):
            for layer in model.layers:
                layer.attention.paged_attention_config = self.paged_attention_config
                # Reinitialize KV cache with paged attention config
                # This will create paged attention KV cache: [max_num_blocks, n_local_kv_heads, block_size, head_dim]
                layer.attention.init_kv_cache(
                    self.model_args[model_idx], self.model_args[model_idx].weight_cache_path(dtype)
                )

        # Get tokenizer and processor from model_args
        self.tokenizer = self.model_args[0].tokenizer
        self.processor = self.model_args[0].processor

        # Create generator for running inference
        self.generator = Generator(
            self.model,
            self.model_args,
            self.device,
            processor=self.processor,
            tokenizer=self.tokenizer,
        )

        # Get KV cache from attention layers (paged attention)
        # Each layer has layer_past which contains [k_cache, v_cache] for paged attention
        self._kv_cache = [[layer.attention.layer_past for layer in model.layers] for model in self.model]

        self._is_initialized = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for embedding generation (prefill-only).

        This is the main interface method called by vLLM for embedding inference.
        Unlike generation models, embedding models only have prefill (no decode step).
        vLLM detects this by checking that the model has forward() but not
        prefill_forward()/decode_forward(), and routes it to TTModelRunnerPooling
        which calls this forward() method directly.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask tensor
            token_type_ids: Optional token type IDs tensor (not used for Qwen3)
            position_ids: Optional position IDs tensor

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape
        logger.debug(f"Qwen3-Embedding forward: processing batch_size={batch_size}, seq_len={seq_len}")

        # Ensure batch size and sequence length are within limits
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max {self.max_batch_size}"
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Initialize model if not already done
        # Use max_batch_size for initialization to support full batch capacity
        # The actual batch_size from the request may be smaller, but we need to initialize
        # with max_batch_size to support future requests up to that size
        self._initialize_model(self.max_batch_size, self.max_seq_len)

        # Pad sequence length to max_seq_len if needed
        original_seq_len = seq_len  # Keep original for attention mask
        if input_ids.shape[1] < self.max_seq_len:
            padded_input_ids = torch.zeros([batch_size, self.max_seq_len], dtype=input_ids.dtype)
            padded_input_ids[:, :seq_len] = input_ids
            input_ids = padded_input_ids
            seq_len = self.max_seq_len  # Update seq_len to padded length
        elif input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            seq_len = self.max_seq_len

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, original_seq_len, dtype=torch.float32)
        else:
            # Pad attention mask if needed
            if attention_mask.shape[1] < self.max_seq_len:
                padded_attention_mask = torch.zeros([batch_size, self.max_seq_len], dtype=attention_mask.dtype)
                padded_attention_mask[:, :original_seq_len] = attention_mask
                attention_mask = padded_attention_mask
            elif attention_mask.shape[1] > self.max_seq_len:
                attention_mask = attention_mask[:, : self.max_seq_len]

        # For Qwen3-Embedding, we need to run the model forward pass
        # Since Qwen3-Embedding works through tt-inference-server, we use the
        # standard Transformer model but extract embeddings differently

        # Prepare inputs for prefill
        # Convert input_ids to proper format
        input_tokens_pt = input_ids.view(batch_size, -1)

        # Create page_table for paged attention
        # With paged attention, we always need a page_table
        from models.tt_transformers.tt.common import num_blocks_in_seq

        actual_seq_len = input_ids.shape[1]  # This should be max_seq_len after padding
        block_size = self.paged_attention_config.block_size
        num_blocks = num_blocks_in_seq(actual_seq_len, block_size)

        # Create page_table with shape [batch_size, num_blocks]
        # For paged attention, we need sequential block indices starting from 0
        # Each batch item gets its own set of distinct sequential blocks to avoid conflicts
        # Note: Even though users are processed sequentially, they all use user_id=0 when
        # page_table is present, so they need distinct physical blocks to avoid overwriting
        # each other's KV cache data.
        page_table = torch.zeros(batch_size, num_blocks, dtype=torch.int32)
        for i in range(batch_size):
            # Each sequence gets blocks starting from i * num_blocks to avoid conflicts
            # This ensures batch item 0 uses blocks [0, num_blocks-1],
            # batch item 1 uses blocks [num_blocks, 2*num_blocks-1], etc.
            page_table[i] = torch.arange(i * num_blocks, (i + 1) * num_blocks, dtype=torch.int32)

        logger.debug(
            f"Creating page_table for paged attention: "
            f"seq_len={actual_seq_len}, block_size={block_size}, num_blocks={num_blocks}, "
            f"batch_size={batch_size}, original_seq_len={original_seq_len}"
        )

        # Run prefill to get hidden states (before LM head) for embeddings
        # IMPORTANT: The TT backend processes users sequentially (one at a time) due to
        # architectural constraints. The attention mechanism uses user_id to index into
        # the KV cache, requiring single-user processing. This is expected behavior and
        # not a bug. For batch_size=32, this will process 32 users sequentially.
        #
        # TRACE REUSE: The trace is captured once per (prefill_seq_len, model_id) combination
        # and reused for all subsequent batches with the same sequence length. This means:
        # - First batch: Captures trace (slower, ~1-2 seconds)
        # - Subsequent batches: Reuses trace (faster, ~100-200ms per batch)
        # The trace key is: f"{prefill_seq_len}_{model_id}", where prefill_seq_len comes from
        # get_padded_prefill_len(seq_len). We use the padded seq_len (max_seq_len) for trace
        # reuse, but use original_seq_len for prompt_lens to correctly extract the last real token.
        #
        # CRITICAL: Use original_seq_len for prompt_lens, not padded seq_len!
        # prompt_lens is used to set last_token_idx = seq_len - 1, which must point to the
        # actual last token, not a padding token. The padding is only for trace reuse.
        hidden_states = self.generator.prefill_forward_text(
            input_tokens_pt,
            page_table=page_table,
            kv_cache=self._kv_cache,
            prompt_lens=[original_seq_len] * batch_size,  # Use original_seq_len, not padded seq_len!
            enable_trace=True,  # Explicitly enable trace for best performance
            return_hidden_states=True,  # Return hidden states before LM head, not logits
        )

        # hidden_states shape: [batch_size, hidden_size]
        # This is the last token's hidden state after layer norm, before LM head
        # For Qwen3-Embedding, this is the correct embedding output
        embeddings = hidden_states

        # Ensure output is 2D: [batch_size, embedding_dim]
        if embeddings.dim() == 1 and batch_size == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 2:
            embeddings = embeddings.view(batch_size, -1)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.dim

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
    Register the Qwen3-Embedding model with vLLM's ModelRegistry.

    This function should be called to make the model available to vLLM.
    Typically called from vLLM's model registration code.

    Note: The actual registration happens in tt-vllm-plugin's register_models()
    function, which registers "TTQwen3Model" -> Qwen3ForEmbedding.
    This function is kept for compatibility but may not be called directly.
    """
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        # Register as TTQwen3Model (TT-prefixed version for TT platform)
        ModelRegistry.register_model(
            "TTQwen3Model",
            Qwen3ForEmbedding,
        )
        logger.info("Successfully registered TTQwen3Model (Qwen3-Embedding-8B) with vLLM ModelRegistry")
    except ImportError:
        logger.warning(
            "vLLM ModelRegistry not available. "
            "Make sure vLLM is installed and the model is registered in vLLM's model loader."
        )
