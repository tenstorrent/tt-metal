# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import AttentionConfig


def init_kv_cache(
    mesh_device,
    config: AttentionConfig,
    mesh_config: MeshConfig,
    paged_attention_config=None,
    cache_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
):
    """
    Initialize KV cache for both paged and non-paged attention.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        mesh_config: Mesh parallelization config
        paged_attention_config: Optional paged attention configuration
        cache_dtype: Data type for cache tensors (default: bfloat8_b)
        tensor_cache_path: Optional path for cache file

    Returns:
        List [k_cache, v_cache]
    """
    # Determine cache shape based on paged vs non-paged attention
    kv_cache_repeats = mesh_device.shape[0] if config.users_row_sharded else 1
    if paged_attention_config:
        # Paged attention cache shape: [max_num_blocks, num_kv_heads, block_size, head_dim]
        cache_shape = [
            paged_attention_config.max_num_blocks * kv_cache_repeats,
            config.num_kv_heads // mesh_device.shape[1],
            paged_attention_config.block_size,
            config.head_dim,
        ]
    else:
        # Standard cache shape: [batch_size, num_kv_heads, max_seq_len, head_dim]
        cache_shape = [
            config.max_local_batch_size * kv_cache_repeats,
            config.num_kv_heads // mesh_device.shape[1],
            config.max_seq_len,
            config.head_dim,
        ]

    # Create K cache
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None))
        if config.users_row_sharded
        else ttnn.ReplicateTensorToMesh(mesh_device)
    )
    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create V cache
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


def get_kv_memory_config(mesh_device, max_local_batch_size: int, num_local_kv_heads: int, head_dim: int):
    """
    Get sharded memory config for KV tensors in decode mode.

    Args:
        mesh_device: TTNN mesh device
        max_local_batch_size: Maximum local batch size per device
        num_local_kv_heads: Number of KV heads per device
        head_dim: Head dimension

    Returns:
        Sharded memory config for KV tensors
    """
    grid_size = ttnn.CoreCoord(8, 8)

    # KV tensors should be [local_batch_size, num_local_kv_heads, 1, head_dim] for decode
    kv_shape = (1, max_local_batch_size, num_local_kv_heads, head_dim)
    kv_shard_height = nearest_y(kv_shape[1], ttnn.TILE_SIZE)  # height = num_local_kv_heads
    kv_shard_width = kv_shape[3]  # width = head_dim
    kv_num_cores = kv_shape[1]  # cores = 1 (sequence length for decode)

    kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, grid_size, row_wise=True)

    return ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, kv_shard_width),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
