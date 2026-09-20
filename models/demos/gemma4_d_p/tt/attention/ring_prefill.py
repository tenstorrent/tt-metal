# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel prefill over chunk-major ring caches.

Each rank stores its local slab of every chunk:
    local row (chunk * L + j) on rank r -> global token (chunk * C + r * L + j)
where C is the global chunk size and L = C / CP.

Ring SDPA gathers the cached prefix internally and applies causal attention.
Local layers also apply the sliding window. Every chunk uses this path.
"""

from dataclasses import dataclass

import ttnn

from .global_kv_cache import GLOBAL_HEAD_DIM, GLOBAL_PACKED_DIM, GLOBAL_ROTARY_DIM

TILE_HEIGHT = 32


@dataclass(frozen=True)
class SlidingRingKVCache:
    """Separate K and V cache tensors for sliding attention."""

    k: ttnn.Tensor
    v: ttnn.Tensor


@dataclass(frozen=True)
class GlobalRingKVCache:
    """One physical global-attention cache with overlapping K and V views."""

    kv: ttnn.Tensor


def migration_ring_memory_config(mesh_device, row_dim):
    """One migratable 32-token row per round-robin DRAM shard."""
    banks = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0))
            for bank in range(mesh_device.dram_grid_size().x)
        ]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, TILE_HEIGHT, row_dim],
            grid=banks,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _allocate_migration_ring_cache(mesh_device, shape, dtype, row_dim):
    """Allocate and zero a replicated mesh buffer without a host-sized staging tensor."""
    from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

    cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mesh_device, migration_ring_memory_config(mesh_device, row_dim)
    )
    DRAMZeroFill.op(cache)
    dist_shape = ttnn.MeshShape(*tuple(mesh_device.shape))
    coords = [
        ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())]) for coord in ttnn.MeshCoordinateRange(dist_shape)
    ]
    cache.update_tensor_topology(
        ttnn.TensorTopology(dist_shape, [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], coords)
    )
    return cache


def ring_cache_capacity(max_seq_len, prefill_chunk_size):
    """Reserve two chunks so sliding SDPA always selects its chunked implementation."""
    return max(max_seq_len, 2 * prefill_chunk_size)


def ring_cache_seq_len(max_seq_len, cp):
    """Per-rank cache sequence length. Each rank stores 1/cp of every chunk."""
    assert max_seq_len % cp == 0, f"max_seq_len {max_seq_len} must be divisible by CP degree {cp}"
    return max_seq_len // cp


def init_sliding_ring_kv_cache(
    mesh_config, num_local_kv_heads, head_dim, max_seq_len, num_layers=1, num_users=1, cache_dtype=ttnn.bfloat8_b
):
    """Contiguous CP-sharded K/V caches for the ring path.

    Shape per rank is ``[num_users*num_layers, num_local_kv_heads, max_seq_len/cp,
    head_dim]``. The batch dim packs users and layers user-major
    (``slot = user*num_layers + layer``), which is how ``update_padded_kv_cache``
    indexes it.

    Sequence is sharded across the CP axis and heads are already TP-local, so the
    mapper only needs to split the sequence: every rank allocates the same shape and
    the contents diverge on first write.

    bfloat8_b because ring_joint requires BFP8_B K/V (BF16 Q).
    """
    mesh_device = mesh_config.device
    cp = mesh_config.cp_degree
    seq_local = ring_cache_seq_len(max_seq_len, cp)
    shape = [num_users * num_layers, num_local_kv_heads, seq_local, head_dim]

    def _zeros():
        # Every rank holds an identically shaped slab; content diverges on first write.
        return _allocate_migration_ring_cache(mesh_device, shape, cache_dtype, head_dim)

    return SlidingRingKVCache(k=_zeros(), v=_zeros())


def init_global_ring_kv_cache(
    mesh_config, num_local_kv_heads, max_seq_len, num_layers=1, num_users=1, cache_dtype=ttnn.bfloat8_b
):
    """Allocate the global [Krot128 | Vordered512] CP-sharded cache."""
    mesh_device = mesh_config.device
    cp = mesh_config.cp_degree
    seq_local = ring_cache_seq_len(max_seq_len, cp)
    shape = [num_users * num_layers, num_local_kv_heads, seq_local, GLOBAL_PACKED_DIM]
    cache = _allocate_migration_ring_cache(mesh_device, shape, cache_dtype, GLOBAL_PACKED_DIM)
    return GlobalRingKVCache(cache)


def write_chunk_to_global_ring_cache(
    cache,
    chunk,
    mesh_config,
    kv_actual_global,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
    prefill_metadata=None,
    actual_end=None,
):
    """Append one packed global-attention chunk to its CP-local history."""

    original_chunk = chunk

    if chunk.dtype != cache.dtype:
        chunk = ttnn.typecast(chunk, cache.dtype)

    if prefill_metadata is not None:
        slot_idx_t = prefill_metadata.slot_idx
        kv_actual_global_t = prefill_metadata.kv_actual_global

        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache=cache,
            input=chunk,
            slot_idx=slot_idx_t,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual_global_t,
            valid_global=prefill_metadata.actual_end,
            cluster_axis=mesh_config.cp_axis,
        )
    else:
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache=cache,
            input=chunk,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual_global,
            valid_global=actual_end,
            cluster_axis=mesh_config.cp_axis,
        )

    if chunk is not original_chunk:
        chunk.deallocate(True)


def global_ring_prefill_attention(
    tt_q,
    cache_kv,
    mesh_config,
    ccl_manager,
    prefill_metadata,
    num_local_kv_heads,
    max_seq_len,
    logical_n,
    kv_actual_global,
    scale=1.0,
    compute_kernel_config=None,
    program_config=None,
    layer_idx=0,
    num_layers=1,
):
    """Attend from two transient logical views of the single packed cache."""
    cache_shape = tuple(cache_kv.shape)
    cache_k = ttnn.slice(
        cache_kv, (0, 0, 0, 0), cache_shape[:-1] + (GLOBAL_HEAD_DIM,), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cache_v = ttnn.slice(
        cache_kv,
        (0, 0, 0, GLOBAL_ROTARY_DIM),
        cache_shape[:-1] + (GLOBAL_PACKED_DIM,),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = _ring_prefill_attention(
        tt_q,
        cache_k,
        cache_v,
        mesh_config,
        ccl_manager,
        prefill_metadata,
        num_local_kv_heads,
        GLOBAL_HEAD_DIM,
        max_seq_len,
        logical_n,
        kv_actual_global,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
        layer_idx=layer_idx,
        num_layers=num_layers,
    )
    cache_k.deallocate(True)
    cache_v.deallocate(True)
    return out


def ring_prefill_program_config(mesh_device, ccl_manager, head_dim, q_chunk_size, k_chunk_size):
    """SDPA program config for the ring path.

    The compute grid must exclude the CCL column that ``ccl_core_grid_offset``
    points at — ring_joint asserts the CCL and SDPA core sets are disjoint.

    q/k chunk sizes are the ones the chunked sliding path accepts (q in {64,128},
    k == 128); k_chunk also sets the halo granularity, since the halo is the
    window rounded up to whole k chunks.
    """
    grid = ccl_manager.compute_grid_size
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )


def write_chunk_to_sliding_ring_cache(
    cache_k,
    cache_v,
    tt_k,
    tt_v,
    mesh_config,
    kv_actual_global,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
    prefill_metadata=None,
    actual_end=None,
):
    """Append one sliding-attention chunk (K and V) to its CP-local history."""

    for cache, chunk in ((cache_k, tt_k), (cache_v, tt_v)):
        original_chunk = chunk

        if chunk.dtype != cache.dtype:
            chunk = ttnn.typecast(chunk, cache.dtype)

        if prefill_metadata is not None:
            slot_idx_t = prefill_metadata.slot_idx
            kv_actual_global_t = prefill_metadata.kv_actual_global

            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=cache,
                input=chunk,
                slot_idx=slot_idx_t,
                layer_idx=layer_idx,
                num_layers=num_layers,
                kv_actual_global=kv_actual_global_t,
                valid_global=prefill_metadata.actual_end,
                cluster_axis=mesh_config.cp_axis,
            )
        else:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=cache,
                input=chunk,
                slot_idx=slot_idx,
                layer_idx=layer_idx,
                num_layers=num_layers,
                kv_actual_global=kv_actual_global,
                valid_global=actual_end,
                cluster_axis=mesh_config.cp_axis,
            )

        if chunk is not original_chunk:
            chunk.deallocate(True)


def sliding_ring_prefill_attention(
    tt_q,
    cache_k,
    cache_v,
    mesh_config,
    ccl_manager,
    prefill_metadata,
    num_local_kv_heads,
    head_dim,
    max_seq_len,
    logical_n,
    kv_actual_global,
    sliding_window_size=None,
    scale=1.0,
    compute_kernel_config=None,
    program_config=None,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
):
    """Attend sliding layers using separate K and V ring caches."""
    return prefill_metadata.sliding.attention(
        attention_fn=_ring_prefill_attention,
        tt_q=tt_q,
        cache_k=cache_k,
        cache_v=cache_v,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        prefill_metadata=prefill_metadata,
        num_local_kv_heads=num_local_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        logical_n=logical_n,
        kv_actual_global=kv_actual_global,
        sliding_window_size=sliding_window_size,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
        layer_idx=layer_idx,
        num_layers=num_layers,
        slot_idx=slot_idx,
    )


def _ring_prefill_attention(
    tt_q,
    cache_k,
    cache_v,
    mesh_config,
    ccl_manager,
    prefill_metadata,
    num_local_kv_heads,
    head_dim,
    max_seq_len,
    logical_n,
    kv_actual_global,
    sliding_window_size=None,
    scale=1.0,
    compute_kernel_config=None,
    program_config=None,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
):
    """Attend this rank's Q shard over the whole cached prefix, via the CP ring.

    ``logical_n`` fixes the cache capacity at capture. Device metadata supplies
    the valid prefix on each replay; ``kv_actual_global`` is the prefix before
    this chunk when metadata is updated here.

    Returns ``[1, num_local_q_heads, q_local, head_dim]`` — this rank's rows only, so
    the output stays CP-sharded exactly like the input.
    """
    mesh_device = mesh_config.device
    if program_config is None:
        # Utilization testing identified these as the best-performing chunk sizes.
        _q_chunk, _k_chunk = (128, 128) if sliding_window_size else (96, 256)
        program_config = ring_prefill_program_config(
            mesh_device,
            ccl_manager,
            head_dim,
            q_chunk_size=_q_chunk,
            k_chunk_size=_k_chunk,
        )
    cp = mesh_config.cp_degree
    cache_seq = ring_cache_seq_len(max_seq_len, cp)

    # Buffer size depends on the mode, and the two requirements are opposites.
    #
    # Dense (no window): ring_joint gathers the entire per-device shard, so the buffer
    # must span the FULL cache capacity — not logical_n, which survives a 2-chunk run
    # and then fails "gather dim 2 too small".
    #
    # Sliding: only the predecessor halo is exchanged, and the op *requires* a compact
    # buffer (gathered rows < cache_seq * ring), rejecting a full-capacity one with
    # "requires a compact halo buffer". Size it to the halo, which is the window
    # rounded up to whole k chunks.
    if sliding_window_size:
        k_chunk = program_config.k_chunk_size
        halo_tokens = -(-(sliding_window_size - 1) // k_chunk) * k_chunk
        gather_seq = max(halo_tokens, TILE_HEIGHT)
    else:
        gather_seq = cache_seq * cp
    buffer_k = ccl_manager.get_ring_gather_buffer(
        "ring_k", num_local_kv_heads, gather_seq, head_dim, cache_k.dtype, cache_k.memory_config()
    )
    buffer_v = ccl_manager.get_ring_gather_buffer(
        "ring_v", num_local_kv_heads, gather_seq, head_dim, cache_v.dtype, cache_v.memory_config()
    )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        persistent_output_buffer_k=buffer_k,
        persistent_output_buffer_v=buffer_v,
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=mesh_config.cp_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ttnn.CoreCoord(*ccl_manager.ring_attention_ccl_core_grid_offset),
        use_column_major_ccl=True,
        is_causal=True,
        is_balanced=False,
        slot_id=prefill_metadata.slot_idx,
        kv_actual_isl_tensor=prefill_metadata.kv_actual_global,
        kv_cache_num_layers=num_layers,
        kv_cache_layer_idx=layer_idx,
        sliding_window_size=sliding_window_size,
    )
    return out
