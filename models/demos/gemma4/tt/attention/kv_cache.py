# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV cache initialization for Gemma4 attention with TP support.

Per-device cache uses local KV head count (num_kv_heads // tp).
Follows gpt-oss kv_cache.py pattern.
"""

import torch

import ttnn
from models.demos.gemma4.utils.general_utils import get_cache_file_name


def init_kv_cache(
    mesh_device,
    config,
    max_batch_size=1,
    max_seq_len=131072,
    paged_attention_config=None,
    cache_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
):
    """
    Initialize KV cache for a single attention layer.

    For TP > 1, each device gets num_kv_heads // tp heads (column-parallel sharding).
    The cache tensor is replicated to each device with the local head count.

    Args:
        mesh_device: TT device or mesh device
        config: Gemma4AttentionConfig for this layer
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        paged_attention_config: Optional paged attention config
        cache_dtype: Cache tensor dtype
        tensor_cache_path: Optional cache file path

    Returns:
        [k_cache, v_cache] list of TT tensors
    """
    # Determine TP from mesh shape (column axis)
    is_mesh = hasattr(mesh_device, "shape")
    tp = mesh_device.shape[1] if is_mesh else 1

    num_local_kv_heads = config.num_key_value_heads // tp
    head_dim = config.head_dim

    if paged_attention_config:
        cache_shape = [
            paged_attention_config.max_num_blocks,
            num_local_kv_heads,
            paged_attention_config.block_size,
            head_dim,
        ]
    else:
        cache_shape = [
            max_batch_size,
            num_local_kv_heads,
            max_seq_len,
            head_dim,
        ]

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    v_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"v_cache_{cache_shape}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return [k_cache, v_cache]
