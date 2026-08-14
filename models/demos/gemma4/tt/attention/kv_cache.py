# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV cache initialization for Gemma4 attention with TP support.

Per-device cache uses local KV head count (num_kv_heads // tp).
Follows gpt-oss kv_cache.py pattern.
"""

import torch

import ttnn
from models.demos.gemma4.tt.precision import dtype_to_str
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# Hot cache blocks held in staging per decode slot for the loop-free packed KV
# write: a P-token speculative tail (P <= block_size) straddles at most 2 pages,
# so each slot reserves 2 staging blocks (cur_pos block + spill). Must match the
# spec-decode driver's staging bookkeeping (see tt/spec_decode.py).
PV_HOT_BLOCKS = 2


def init_kv_cache(
    mesh_device,
    config,
    max_batch_size=1,
    max_seq_len=131072,
    paged_attention_config=None,
    cache_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
    max_num_blocks_override=None,
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
        max_num_blocks_override: When set (only meaningful in paged mode), size the
            physical block pool to this value instead of paged_attention_config.max_num_blocks.
            vLLM's hybrid kv_cache_groups uses this for SlidingWindowSpec layers: the
            sliding-window cache holds only sliding_window/block_size blocks per sequence,
            while the per-layer page_table is zero-padded to max_model_len/block_size.

    Returns:
        [k_cache, v_cache] list of TT tensors
    """
    # Determine TP from mesh shape (column axis)
    is_mesh = hasattr(mesh_device, "shape")
    tp = mesh_device.shape[1] if is_mesh else 1

    # When KV heads < TP, each device gets 1 KV head (GQA-assigned, not all heads)
    num_local_kv_heads = 1 if config.num_key_value_heads < tp else config.num_key_value_heads // tp
    head_dim = config.head_dim

    if paged_attention_config:
        max_num_blocks = (
            max_num_blocks_override if max_num_blocks_override is not None else paged_attention_config.max_num_blocks
        )
        cache_shape = [
            max_num_blocks,
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

    # Tag the filename with the dtype, as the weight tensors do. as_tensor returns a cache hit
    # AS STORED, ignoring the dtype it was asked for, so a name that omits it would hand back a
    # bf16 cache after the precision override moved K/V to bfp8 -- silently, and only detectable
    # downstream where SDPA expects the narrower format.
    dtype_suffix = f"_{dtype_to_str(cache_dtype)}"

    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape}{dtype_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    v_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"v_cache_{cache_shape}{dtype_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return [k_cache, v_cache]


def init_kv_staging(
    mesh_device,
    config,
    max_batch_size,
    block_size,
    blk=PV_HOT_BLOCKS,
    cache_dtype=ttnn.bfloat16,
):
    """Per-layer staging buffers for the loop-free packed-verify KV write.

    Holds, per decode slot, the ``blk`` "hot" cache blocks it is currently
    appending into (the partial block at cur_pos + one spill block). The
    packed-verify write merges this resident copy with the step's new K/V and
    re-fills the committed cache from it — so the committed cache is never read
    on the hot path (see ``decode.py::_packed_fill_kv_loopfree_embed``).

    Shape ``[1, num_local_kv_heads, max_batch_size*blk*block_size, head_dim]``
    in TILE/DRAM — same seq layout as the ``paged_fill_cache`` input. Slot ``s``
    owns seq positions ``[s*blk*block_size, (s+1)*blk*block_size)``; block-slot
    0 is the cur_pos block, block-slot 1 the spill block.

    Returns ``[k_staging, v_staging]``.
    """
    is_mesh = hasattr(mesh_device, "shape")
    tp = mesh_device.shape[1] if is_mesh else 1
    num_local_kv_heads = 1 if config.num_key_value_heads < tp else config.num_key_value_heads // tp
    stage_shape = [1, num_local_kv_heads, max_batch_size * blk * block_size, config.head_dim]
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    def _zeros():
        return ttnn.as_tensor(
            torch.zeros(stage_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=cache_dtype,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return [_zeros(), _zeros()]


def export_paged_kv_cache_natural_order(cache, mesh_device, mesh_config, block_size):
    """Read a context-parallel paged cache back with the CP permutation undone.

    This is the "package it for the decode side" step. Under CP the block pool is
    sharded along the CP axis, so a token's location is implied by
    ``(cp_rank, local_block, row)`` rather than by a global block id:

        global token = cp_rank * tokens_per_rank + local_block * block_size + row

    Prefill can leave it that way — a rank's shard is exactly what it computed, so
    the fill needs no gather (see ``Gemma4Model._cp_block_pool_override``). Whoever
    consumes the cache does not share that layout, so the permutation is undone here.

    Undoes **CP only**. Tensor-parallel head sharding is preserved as the leading
    axis, because a disaggregated decode target is itself TP-sharded and wants its
    heads already split; it is the sequence axis that has to become contiguous.

    Returns a torch tensor ``[tp, num_tokens_global, num_local_kv_heads, head_dim]``.
    With CP off this is just the per-column cache flattened, so callers can use it
    unconditionally.
    """
    from models.demos.gemma4.tt.ccl import cp_degree

    rows, cols = tuple(mesh_device.shape)
    cp = cp_degree(mesh_config) if mesh_config is not None else 1
    shards = ttnn.get_device_tensors(cache)

    # CP lives on sp_axis; the other axis is TP. Only sp_axis == 0 is exercised
    # (see MeshConfig.tp_axis), so refuse the transpose rather than guess at it.
    if mesh_config is not None and cp > 1 and mesh_config.sp_axis != 0:
        raise NotImplementedError(f"export expects CP on mesh axis 0, got sp_axis={mesh_config.sp_axis}")

    per_column = []
    for c in range(cols):
        # Walk CP ranks in order so the concatenated sequence is globally ordered.
        rank_chunks = []
        for r in range(cp if cp > 1 else 1):
            shard = ttnn.to_torch(shards[r * cols + c]).float()
            # [blocks_local, kv_local, block_size, head_dim] -> [tokens_local, kv_local, head_dim]
            blocks_local, kv_local, bs, head_dim = shard.shape
            assert bs == block_size, f"cache block_size {bs} != {block_size}"
            tokens = shard.permute(0, 2, 1, 3).reshape(blocks_local * bs, kv_local, head_dim)
            rank_chunks.append(tokens)
        per_column.append(torch.cat(rank_chunks, dim=0) if len(rank_chunks) > 1 else rank_chunks[0])

    return torch.stack(per_column, dim=0)
