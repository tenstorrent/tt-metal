# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
KV-cache implementation for Qwen3-TTS decode mode.

Provides pre-allocated KV caches for efficient autoregressive decoding.
"""

from typing import List, Tuple

import torch

import ttnn


class KVCache:
    """
    Key-Value cache for transformer attention layers.

    Stores K and V tensors for each layer to enable efficient autoregressive decoding.
    """

    def __init__(
        self,
        device,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ):
        """
        Initialize KV cache.

        Args:
            device: TTNN device
            num_layers: Number of transformer layers
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            num_kv_heads: Number of key-value heads
            head_dim: Dimension of each head
            dtype: Data type for cache tensors
        """
        self.device = device
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Current sequence length for each batch
        self.seq_len = 0

        # Pre-allocate cache tensors for each layer
        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        self.k_cache = []
        self.v_cache = []

        for _ in range(num_layers):
            k = ttnn.from_torch(
                torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v = ttnn.from_torch(
                torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.k_cache.append(k)
            self.v_cache.append(v)

    def update(self, layer_idx: int, k: ttnn.Tensor, v: ttnn.Tensor, start_pos: int):
        """
        Update the KV cache for a specific layer.

        Args:
            layer_idx: Layer index
            k: New key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: New value tensor [batch, num_kv_heads, seq_len, head_dim]
            start_pos: Starting position in the sequence

        Returns:
            Tuple of (cached_k, cached_v) with full sequence
        """
        seq_len = k.shape[2]

        # Fill cache at the specified position
        # Use update_cache with update_idx for positional updates
        ttnn.update_cache(self.k_cache[layer_idx], k, update_idx=start_pos)
        ttnn.update_cache(self.v_cache[layer_idx], v, update_idx=start_pos)

        self.seq_len = start_pos + seq_len

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get(self, layer_idx: int):
        """
        Get cached K and V for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (k_cache, v_cache) tensors
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        """Reset the cache (clear all stored values)."""
        self.seq_len = 0
        for i in range(self.num_layers):
            # Zero out the cache
            self.k_cache[i] = ttnn.multiply(self.k_cache[i], 0.0)
            self.v_cache[i] = ttnn.multiply(self.v_cache[i], 0.0)


def create_kv_cache(
    device,
    config,
    max_batch_size: int = 1,
    max_seq_len: int = 2048,
) -> KVCache:
    """
    Factory function to create KV cache for a model config.

    Args:
        device: TTNN device
        config: Model configuration (Talker or CodePredictor config)
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length

    Returns:
        Initialized KVCache
    """
    return KVCache(
        device=device,
        num_layers=config.num_hidden_layers,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )


def create_kv_cache_list(
    device,
    config,
    max_batch_size: int = 1,
    max_seq_len: int = 2048,
    dtype=ttnn.bfloat16,
) -> List[Tuple[ttnn.Tensor, ttnn.Tensor]]:
    """
    Create KV caches as a list of (k_cache, v_cache) tuples for each layer.

    This format is compatible with the updated attention module.

    Args:
        device: TTNN device
        config: Model configuration (Talker or CodePredictor config)
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        dtype: Data type for cache tensors

    Returns:
        List of (k_cache, v_cache) tuples, one per layer
    """
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    kv_caches = []
    for _ in range(num_layers):
        k = ttnn.from_torch(
            torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.from_torch(
            torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv_caches.append((k, v))

    return kv_caches
