# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only **paged** KV cache for disaggregated GPT-OSS prefill (`gpt_oss_d_p`).

Bridges ``kv_cache_prefill_only.py`` (outer SP mesh upload, migration slots) and
``kv_cache.py`` (paged block layout for ``paged_fill_cache`` / chunked SDPA):

  * shape per device:
      ``[num_blocks_local, num_kv_heads // TP, block_size, head_dim]``
  * ``PagedAttentionConfig`` via ``make_paged_attention_config_for_prefill``
  * ``ShardTensor2dMesh(dims=(sp_axis, None))``: each SP row holds a block shard
    covering ``max_seq_len // SP`` tokens (ownership-aligned page tables)
  * ``max_local_batch_size`` users via ``paged_fill_cache(..., batch_idx=slot)``
  * **Production:** ``make_paged_kv_memory_config()`` → INTERLEAVED DRAM (required by
    ``paged_fill_cache`` / ``chunked_scaled_dot_product_attention``)

For chunked prefill (``CHUNKED_PREFILL_PLAN.md``): allocate once at pipeline build;
Each chunk forward calls ``paged_fill_cache`` with a per-row ``chunk_page_table``
slice. Migration address tables: ``utils/kv_cache_table_paged.py`` (INTERLEAVED page
indices; readback via ``read_paged_device_chunk``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.kv_cache_table import CHUNK_N_TOKENS
from models.tt_transformers.tt.common import PagedAttentionConfig

from .config import AttentionConfig
from .kv_cache_prefill_only import make_gpt_oss_prefill_kv_memory_config

# GPT-OSS disaggregated prefill default (see PREFILL_PROPOSAL.md / migration_setup.py).
DEFAULT_PREFILL_PAGE_BLOCK_SIZE = 64


def make_paged_kv_memory_config() -> ttnn.MemoryConfig:
    """INTERLEAVED DRAM — required by ``paged_fill_cache`` and chunked SDPA on the cache tensor."""
    return ttnn.DRAM_MEMORY_CONFIG


def make_paged_kv_memory_config_for_migration_table(head_dim: int) -> ttnn.MemoryConfig:
    """NdShardSpec DRAM for ``kv_cache_table_paged`` device tests only.

    Not valid for ``paged_fill_cache`` (which requires INTERLEAVED). Do not use in the
    prefill runner / chunked SDPA path.
    """
    return make_gpt_oss_prefill_kv_memory_config(head_dim)


@dataclass(frozen=True)
class PrefillPagedKvCacheSetup:
    """Bundle passed to ``init_kv_cache`` for production chunked prefill + paged SDPA."""

    paged_attention_config: PagedAttentionConfig
    memory_config: ttnn.MemoryConfig


def make_prefill_paged_kv_cache_setup(
    mesh_device,
    *,
    max_seq_len: int,
    max_local_batch_size: int = 1,
    sp_axis: int = 0,
    block_size: int = DEFAULT_PREFILL_PAGE_BLOCK_SIZE,
    sp_shard_blocks: bool = True,
) -> PrefillPagedKvCacheSetup:
    """Build ``PagedAttentionConfig`` + INTERLEAVED memory config for one prefill mesh.

    Example (prefill runner / pipeline)::

        setup = make_prefill_paged_kv_cache_setup(
            mesh_device,
            max_seq_len=131072,
            max_local_batch_size=num_migration_slots,
        )
        k_cache, v_cache = init_kv_cache(
            mesh_device,
            attention_config,
            setup.paged_attention_config,
            memory_config=setup.memory_config,
        )
    """
    sp = mesh_device.shape[sp_axis]
    return PrefillPagedKvCacheSetup(
        paged_attention_config=make_paged_attention_config_for_prefill(
            max_seq_len=max_seq_len,
            max_local_batch_size=max_local_batch_size,
            sp=sp,
            block_size=block_size,
            sp_shard_blocks=sp_shard_blocks,
        ),
        memory_config=make_paged_kv_memory_config(),
    )


def blocks_per_sequence(seq_len: int, block_size: int) -> int:
    """Logical page-table blocks needed to cover ``seq_len`` tokens."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return math.ceil(seq_len / block_size)


def blocks_per_sp_shard(max_seq_len: int, sp: int, block_size: int) -> int:
    """Blocks covering one SP row's global token slice ``max_seq_len // sp``."""
    if max_seq_len % sp != 0:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by SP ({sp})")
    return blocks_per_sequence(max_seq_len // sp, block_size)


def make_paged_attention_config_for_prefill(
    max_seq_len: int,
    max_local_batch_size: int = 1,
    sp: int = 1,
    *,
    block_size: int = DEFAULT_PREFILL_PAGE_BLOCK_SIZE,
    sp_shard_blocks: bool = True,
) -> PagedAttentionConfig:
    """Build ``PagedAttentionConfig`` sized for prefill-only paged KV.

    When ``sp_shard_blocks`` is True (default), ``max_num_blocks`` is per device and
    covers one SP row's token slice × ``max_local_batch_size`` slots. When False, each
    device gets the full ``max_seq_len`` block count (replicated mesh upload).
    """
    if block_size % CHUNK_N_TOKENS != 0:
        raise ValueError(
            f"block_size ({block_size}) must be a multiple of migration chunk "
            f"granularity CHUNK_N_TOKENS={CHUNK_N_TOKENS}"
        )
    if sp_shard_blocks:
        blocks_per_row = blocks_per_sp_shard(max_seq_len, sp, block_size)
    else:
        blocks_per_row = blocks_per_sequence(max_seq_len, block_size)
    max_num_blocks = max_local_batch_size * blocks_per_row
    return PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)


def _validate_paged_config(
    config: AttentionConfig,
    paged_attention_config: PagedAttentionConfig,
    sp: int,
    *,
    sp_shard_blocks: bool,
) -> None:
    if paged_attention_config is None:
        raise ValueError("paged_attention_config is required for kv_cache_prefill_only_paged")
    block_size = paged_attention_config.block_size
    if block_size % CHUNK_N_TOKENS != 0:
        raise ValueError(f"block_size ({block_size}) must be a multiple of CHUNK_N_TOKENS={CHUNK_N_TOKENS}")
    if config.max_local_batch_size < 1:
        raise ValueError(f"max_local_batch_size ({config.max_local_batch_size}) must be >= 1")
    if config.max_seq_len % sp != 0:
        raise ValueError(f"max_seq_len ({config.max_seq_len}) must be divisible by SP ({sp})")

    if sp_shard_blocks:
        min_blocks = config.max_local_batch_size * blocks_per_sp_shard(config.max_seq_len, sp, block_size)
    else:
        min_blocks = config.max_local_batch_size * blocks_per_sequence(config.max_seq_len, block_size)
    if paged_attention_config.max_num_blocks < min_blocks:
        raise ValueError(
            f"paged_attention_config.max_num_blocks ({paged_attention_config.max_num_blocks}) "
            f"is smaller than required minimum ({min_blocks}) for "
            f"max_seq_len={config.max_seq_len}, sp={sp}, sp_shard_blocks={sp_shard_blocks}, "
            f"max_local_batch_size={config.max_local_batch_size}, block_size={block_size}"
        )


def init_kv_cache(
    mesh_device,
    config: AttentionConfig,
    paged_attention_config: PagedAttentionConfig,
    sp_axis: int = 0,
    tp_axis: int = 1,
    *,
    sp_shard_blocks: bool = True,
    memory_config: ttnn.MemoryConfig | None = None,
    cache_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
):
    """
    Allocate zero-initialized paged K/V cache tensors for prefill-only serving.

    Args:
        mesh_device: TTNN mesh device.
        config: Attention configuration (``max_seq_len``, ``max_local_batch_size``, heads).
        paged_attention_config: Block size and per-device block count (see
            ``make_paged_attention_config_for_prefill`` or ``make_prefill_paged_kv_cache_setup``).
        sp_axis: Mesh axis for outer sequence parallel (default row axis 0).
        tp_axis: Mesh axis for tensor parallel (default col axis 1).
        sp_shard_blocks: If True, each SP row's cache covers ``max_seq_len // SP`` tokens
            (matches ``kv_cache_prefill_only`` ownership). If False, each device holds
            blocks for the full ``max_seq_len`` (replicated across SP rows).
        memory_config: Defaults to ``make_paged_kv_memory_config()`` (INTERLEAVED DRAM for
            ``paged_fill_cache`` / chunked SDPA). Override only for migration-table tests.
        cache_dtype: KV dtype (default bfloat8_b).
        tensor_cache_path: Optional weight/cache file prefix.

    Returns:
        ``[k_cache, v_cache]`` — per device:
        ``[num_blocks_local, num_kv_heads // TP, block_size, head_dim]`` where
        ``num_blocks_local == paged_attention_config.max_num_blocks``.
    """
    sp = mesh_device.shape[sp_axis]
    tp = mesh_device.shape[tp_axis]
    num_local_kv_heads = config.num_kv_heads // tp
    if config.num_kv_heads % tp != 0:
        raise ValueError(
            f"num_kv_heads ({config.num_kv_heads}) must be divisible by TP ({tp}), "
            f"got num_local_kv_heads={num_local_kv_heads}"
        )

    _validate_paged_config(config, paged_attention_config, sp, sp_shard_blocks=sp_shard_blocks)

    block_size = paged_attention_config.block_size
    num_blocks_local = paged_attention_config.max_num_blocks
    cache_shape_local = [
        num_blocks_local,
        num_local_kv_heads,
        block_size,
        config.head_dim,
    ]

    if sp_shard_blocks:
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
    else:
        torch_cache_shape = cache_shape_local
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    cache_suffix = f"{cache_shape_local}_sp{sp}_b{block_size}_{'sp_shard' if sp_shard_blocks else 'repl'}"
    kv_mem_config = memory_config if memory_config is not None else make_paged_kv_memory_config()

    k_cache = ttnn.as_tensor(
        torch.zeros(torch_cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"paged_k_cache_{cache_suffix}"),
        memory_config=kv_mem_config,
    )

    v_cache = ttnn.as_tensor(
        torch.zeros(torch_cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"paged_v_cache_{cache_suffix}"),
        memory_config=kv_mem_config,
    )

    return [k_cache, v_cache]
