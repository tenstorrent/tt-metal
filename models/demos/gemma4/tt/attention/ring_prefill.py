# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-chunk prefill attention under context parallelism, via ring_joint SDPA.

Why this exists
---------------
Gemma4's CP-sharded KV cache holds only the tokens each rank computed, which is
what makes the fill free (no gather, no per-device write offset — see
``Gemma4Model._cp_block_pool_override``). The cost is that at chunk > 0 a rank
cannot read history from its own shard: it owns a strided subset of the prefix.

``ring_joint_scaled_dot_product_attention`` resolves that. It reads the CP-sharded
cache and gathers the accumulated prefix around the CP ring *internally*, with
online softmax, so no explicit AllGather materializes the full history. That is
the same mechanism minimax_m3 uses for its dense GQA layers.

Layout
------
The cache is contiguous per rank and chunk-major, matching both
``update_padded_kv_cache``'s write offset and the layout ring_joint reads:

    local row (chunk*L + j) on rank r  <->  global token (chunk*C + r*L + j)

with ``C`` the global chunk size and ``L = C / cp`` the per-rank slab. At a
chunk-aligned boundary the writer's offset math reduces to ``chunk * L`` on every
rank, so no per-device scalar is needed. This is the same permutation Gemma4's
contiguous CP sharding already produces, so activations need no reordering.

Scope
-----
Chunk 0 does not go through here. The chunked path needs a complete predecessor Q
group (``logical_n >= 2 * cp * L``) because the sliding halo wraps onto it, and at
chunk 0 there is none — the op rejects it. Gemma4 keeps using its mask-based CP
path there, which handles a lone windowed chunk directly.

Sliding layers pass ``sliding_window_size``; full-attention layers omit it and get
plain causal attention over the whole prefix. Both otherwise share this path.

The sliding case needs the generalized validation in
``ring_joint_sdpa_device_operation.cpp``: upstream gated the window / head counts /
head_dim to GPT-OSS's values, which Gemma4 does not match.
"""

from dataclasses import dataclass

import ttnn
from models.demos.gemma4.tt.ccl import cp_degree

from .global_kv_cache import GLOBAL_HEAD_DIM, GLOBAL_PACKED_DIM, GLOBAL_ROTARY_DIM

TILE_HEIGHT = 32


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


# Counts ring_joint history reads. The long-context test asserts this advances:
# without it, a silent fallback to the mask path would still produce finite,
# stable-looking output (each chunk attending only within itself), so every
# smoke assertion would pass while history was being ignored.
RING_ATTENTION_CALLS = 0


def reset_ring_attention_calls():
    global RING_ATTENTION_CALLS
    RING_ATTENTION_CALLS = 0


def ring_attention_calls():
    return RING_ATTENTION_CALLS


def ring_cache_seq_len(max_seq_len, cp):
    """Per-rank cache sequence length. Each rank stores 1/cp of every chunk."""
    assert max_seq_len % cp == 0, f"max_seq_len {max_seq_len} must be divisible by CP degree {cp}"
    return max_seq_len // cp


def init_ring_kv_cache(
    mesh_device,
    mesh_config,
    num_local_kv_heads,
    head_dim,
    max_seq_len,
    num_layers=1,
    num_users=1,
    cache_dtype=ttnn.bfloat8_b,
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
    cp = cp_degree(mesh_config)
    seq_local = ring_cache_seq_len(max_seq_len, cp)
    shape = [num_users * num_layers, num_local_kv_heads, seq_local, head_dim]

    def _zeros():
        # Every rank holds an identically shaped slab; content diverges on first write.
        return _allocate_migration_ring_cache(mesh_device, shape, cache_dtype, head_dim)

    return [_zeros(), _zeros()]


@dataclass(frozen=True)
class PackedRingKVCache:
    """One physical global-attention cache with overlapping K and V views."""

    kv: ttnn.Tensor


def init_packed_ring_kv_cache(
    mesh_device,
    mesh_config,
    num_local_kv_heads,
    max_seq_len,
    num_layers=1,
    num_users=1,
    cache_dtype=ttnn.bfloat8_b,
):
    """Allocate the global [Krot128 | Vordered512] CP-sharded cache."""
    cp = cp_degree(mesh_config)
    seq_local = ring_cache_seq_len(max_seq_len, cp)
    shape = [num_users * num_layers, num_local_kv_heads, seq_local, GLOBAL_PACKED_DIM]
    cache = _allocate_migration_ring_cache(mesh_device, shape, cache_dtype, GLOBAL_PACKED_DIM)
    return PackedRingKVCache(cache)


def write_chunk_to_packed_ring_cache(
    cache,
    packed_kv,
    mesh_config,
    kv_actual_global,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
    ccl_manager=None,
):
    """Append one packed global-attention chunk to its CP-local history."""
    chunk = packed_kv if packed_kv.dtype == cache.dtype else ttnn.typecast(packed_kv, cache.dtype)
    if ccl_manager is not None:
        slot_t, kv_t = ccl_manager.get_ring_metadata()
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            chunk,
            slot_t,
            kv_t,
            layer_idx=layer_idx,
            num_layers=num_layers,
            cluster_axis=mesh_config.sp_axis,
        )
    else:
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            chunk,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual_global,
            cluster_axis=mesh_config.sp_axis,
        )
    if chunk is not packed_kv:
        chunk.deallocate(True)


def ring_packed_prefill_attention(
    tt_q,
    cache_kv,
    mesh_device,
    mesh_config,
    ccl_manager,
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
    out = ring_prefill_attention(
        tt_q,
        cache_k,
        cache_v,
        mesh_device,
        mesh_config,
        ccl_manager,
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


def ring_prefill_program_config(mesh_device, ccl_manager, head_dim, q_chunk_size=64, k_chunk_size=128):
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


def write_chunk_to_ring_cache(
    cache_k,
    cache_v,
    tt_k,
    tt_v,
    mesh_config,
    kv_actual_global,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
    ccl_manager=None,
):
    """Write this chunk's per-rank K/V into the CP-sharded cache.

    ``kv_actual_global`` is the *global* prefix length already in the cache before
    this chunk. The writer derives each rank's local row offset from it and from the
    rank's own coordinate along ``cluster_axis``, injected as a runtime arg — which
    is how one mesh-wide program writes a different offset per device. At a
    chunk-aligned boundary that reduces to ``chunk_index * slab``.
    """
    for cache, chunk in ((cache_k, tt_k), (cache_v, tt_v)):
        # The writer requires cache.dtype == input.dtype, and the cache is BFP8_B
        # because that is what ring_joint requires of K/V (with BF16 Q). The model
        # carries K/V in bf16, so cast on the way in.
        if chunk.dtype != cache.dtype:
            chunk = ttnn.typecast(chunk, cache.dtype)
        if ccl_manager is not None:
            # Tensor form: the writer reads slot and prefix length on-device, so the write
            # offset is not baked into runtime args and one captured trace serves every
            # chunk. Same two tensors the ring read uses — they describe the chunk, not
            # the layer, and the host refreshes them once per chunk.
            slot_t, kv_t = ccl_manager.get_ring_metadata()
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                chunk,
                slot_t,
                kv_t,
                layer_idx=layer_idx,
                num_layers=num_layers,
                cluster_axis=mesh_config.sp_axis,
            )
        else:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                chunk,
                slot_idx=slot_idx,
                layer_idx=layer_idx,
                num_layers=num_layers,
                kv_actual_global=kv_actual_global,
                cluster_axis=mesh_config.sp_axis,
            )


def ring_prefill_attention(
    tt_q,
    cache_k,
    cache_v,
    mesh_device,
    mesh_config,
    ccl_manager,
    num_local_kv_heads,
    head_dim,
    max_seq_len,
    logical_n,
    kv_actual_global,
    sliding_window=None,
    scale=1.0,
    compute_kernel_config=None,
    program_config=None,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
):
    """Attend this rank's Q shard over the whole cached prefix, via the CP ring.

    ``logical_n`` is the total valid prefix including this chunk; ``kv_actual_global``
    is the prefix before it. Together they tell the op how much of the cache is real,
    so the not-yet-written tail is masked rather than read as data.

    Returns ``[1, num_local_q_heads, q_local, head_dim]`` — this rank's rows only, so
    the output stays CP-sharded exactly like the input.
    """
    global RING_ATTENTION_CALLS
    RING_ATTENTION_CALLS += 1
    if program_config is None:
        # Global (non-sliding) layers take a wider K chunk. ring_joint SDPA's
        # `q in {64,128}` / `k == 128` allowlist lives inside `if (args.has_sliding_window())`
        # -- it is a structural requirement of the halo, which dense layers do not have. Swept
        # at 32k, per-chunk device time at ring depth 7: k=256 gives 197.8 ms against 201.2 at
        # k=128. q stays 64: it is a true optimum, worse in both directions (214.8 ms at q=32,
        # 221.7 at q=128), and q>=256 overflows L1.
        _k_chunk = 128 if sliding_window else 256
        program_config = ring_prefill_program_config(mesh_device, ccl_manager, head_dim, k_chunk_size=_k_chunk)
    # Shared by every layer; the caller sets them once per chunk via set_ring_metadata.
    metadata = ccl_manager.get_ring_metadata()
    cp = cp_degree(mesh_config)
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
    if sliding_window:
        k_chunk = program_config.k_chunk_size
        halo_tokens = -(-(sliding_window - 1) // k_chunk) * k_chunk
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
        cluster_axis=mesh_config.sp_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ttnn.CoreCoord(*ccl_manager.ring_attention_ccl_core_grid_offset),
        use_column_major_ccl=True,
        is_causal=True,
        # Chunked prefill does not zigzag-balance the causal work.
        is_balanced=False,
        # Per-chunk scalars as metadata tensors rather than Python ints. The readers load
        # them on-device, so they stay out of the program's runtime args and a captured
        # trace replays across chunks; the scalar form would freeze the capturing chunk's
        # prefix length into every replay. The layer packing that kv_cache_batch_idx used
        # to carry moves to kv_cache_num_layers/kv_cache_layer_idx, which the readers
        # combine as slot_id[0]*num_layers + layer_idx — those are constant per layer, so
        # they are safe to keep as host scalars.
        slot_id=metadata[0],
        kv_actual_isl_tensor=metadata[1],
        kv_cache_num_layers=num_layers,
        kv_cache_layer_idx=layer_idx,
        sliding_window_size=sliding_window,
    )
    return out
