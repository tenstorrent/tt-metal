# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only KV cache allocation for disaggregated GPT-OSS prefill (`gpt_oss_d_p`).

Full-sequence prefill (``ttnn_prefill_forward``) with:
  * batch size ``max_local_batch_size`` (one or more users / migration slots per device)
  * non-paged attention (``ttnn.fill_cache``, not ``paged_fill_cache``)
  * SP-sharded KV on mesh rows (``ShardTensor2dMesh(dims=(sp_axis, None))``)
  * ``NdShardSpec`` DRAM (32-token chunks round-robin on 8 banks) — see
    ``make_gpt_oss_prefill_kv_memory_config``; table builder in ``kv_cache_table``
    assumes the same striping for chunk byte addresses
"""

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.kv_cache_table import BH_NUM_DRAM_BANKS, CHUNK_N_TOKENS

from .config import AttentionConfig


def make_gpt_oss_prefill_kv_memory_config(head_dim: int) -> ttnn.MemoryConfig:
    """DRAM memory config for prefill K/V cache — matches DeepSeek ``init_kvpe_cache`` striping."""
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, CHUNK_N_TOKENS, head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )


def init_kv_cache(
    mesh_device,
    config: AttentionConfig,
    sp_axis: int = 0,
    tp_axis: int = 1,
    cache_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
):
    """
    Allocate zero-initialized K and V cache tensors for prefill-only serving.

    Returns ``[k_cache, v_cache]`` with shape per device:
        ``[max_local_batch_size, num_kv_heads // TP, max_seq_len // SP, head_dim]``

    ``max_local_batch_size`` is the number of users (migration slots) sharing the cache;
    ``fill_cache(..., batch_idx=slot)`` and ``KvChunkAddressTable`` slot index use the
    same ``0 .. max_local_batch_size - 1`` range.

    Host upload uses ``ShardTensor2dMesh(dims=(sp_axis, None))``: each SP row holds
    one seq shard; TP columns on that row replicate the same buffer.

    Prefill attention writes via ``ttnn.fill_cache`` at **local** seq indices
    (see ``attention/prefill.py``).
    """
    sp = mesh_device.shape[sp_axis]
    tp = mesh_device.shape[tp_axis]
    num_local_kv_heads = config.num_kv_heads // tp
    if config.num_kv_heads % tp != 0:
        raise ValueError(
            f"num_kv_heads ({config.num_kv_heads}) must be divisible by TP ({tp}), "
            f"got num_local_kv_heads={num_local_kv_heads}"
        )
    if config.max_local_batch_size < 1:
        raise ValueError(f"max_local_batch_size ({config.max_local_batch_size}) must be >= 1 (num_slots / users)")
    if config.max_seq_len % sp != 0:
        raise ValueError(f"max_seq_len ({config.max_seq_len}) must be divisible by SP ({sp})")

    seq_len_local = config.max_seq_len // sp
    cache_shape_local = [
        config.max_local_batch_size,
        num_local_kv_heads,
        seq_len_local,
        config.head_dim,
    ]

    # [SP, batch, heads_local, seq_local, head_dim] for ShardTensor2dMesh on SP rows.
    if sp_axis == 0:
        torch_cache_shape = [sp, *cache_shape_local]
        shard_dims = (0, None)
    elif sp_axis == 1:
        torch_cache_shape = [cache_shape_local[0], sp, *cache_shape_local[1:]]
        shard_dims = (1, None)
    else:
        raise ValueError(f"Unsupported sp_axis={sp_axis} for 2D mesh KV cache")

    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=shard_dims,
    )
    kv_mem_config = make_gpt_oss_prefill_kv_memory_config(config.head_dim)

    k_cache = ttnn.as_tensor(
        torch.zeros(torch_cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape_local}_sp{sp}"),
        memory_config=kv_mem_config,
    )

    v_cache = ttnn.as_tensor(
        torch.zeros(torch_cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"v_cache_{cache_shape_local}_sp{sp}"),
        memory_config=kv_mem_config,
    )

    return [k_cache, v_cache]
