import os
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

from .global_kv_cache import GLOBAL_HEAD_DIM, GLOBAL_PACKED_DIM

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
    """Attend over the packed cache: K is its first GLOBAL_HEAD_DIM columns and V its last GLOBAL_HEAD_DIM."""
    mesh_device = mesh_config.device
    if os.environ.get("G4X_L1DUMP") and not getattr(global_ring_prefill_attention, "_dumped", False):  # LOCAL EXPERIMENT
        global_ring_prefill_attention._dumped = True
        dev = mesh_device.get_devices()[0] if hasattr(mesh_device, "get_devices") else mesh_device
        try:
            print("G4X_L1DUMP view", ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1), flush=True)
        except Exception as e:
            print("G4X_L1DUMP view err", e, flush=True)
        try:
            ttnn.dump_device_memory_state(mesh_device, prefix="g4x_l1dump_")
            print("G4X_L1DUMP dumped", flush=True)
        except Exception as e:
            print("G4X_L1DUMP dump err", e, flush=True)
    if program_config is None:
        sdpa_grid = ccl_manager.compute_grid_size
        q_chunk, k_chunk, k_splits, segmented = ring_sdpa_chunk_sizes(
            tt_q.shape[-2], sliding=False, num_heads=tt_q.shape[1], num_cores=(sdpa_grid.x - 1) * sdpa_grid.y
        )
        program_config = ring_prefill_program_config(
            mesh_device,
            ccl_manager,
            GLOBAL_HEAD_DIM,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            max_k_splits=k_splits,
            # Global attention only: sliding attention gains nothing from LoFi, so it keeps HiFi2.
            matmul_math_fidelity=None if os.environ.get("G4X_GLOBAL_HIFI2") else ttnn.MathFidelity.LoFi,
            segmented_accumulation=segmented,
        )
    # Dense attention gathers each device's whole shard, so the buffer spans the full cache capacity. Sizing it to
    # logical_n survives a 2-chunk run and then fails "gather dim 2 too small".
    cp = mesh_config.cp_degree
    gather_seq = ring_cache_seq_len(max_seq_len, cp) * cp
    # The fused gather's bank-owned schedule needs an interleaved output; the packed cache itself is ND-sharded.
    buffer_kv = ccl_manager.get_ring_gather_buffer(
        "ring_kv", num_local_kv_heads, gather_seq, GLOBAL_PACKED_DIM, cache_kv.dtype, ttnn.DRAM_MEMORY_CONFIG
    )
    # ring_mla rather than ring_joint_scaled_dot_product_attention: only ring_mla reads K and V out of one packed
    # tensor (MLA-style latent V; here Gemma4's [K | V] rows). Both call the same device op,
    # ttnn::prim::ring_joint_scaled_dot_product_attention.
    out, _ = ttnn.transformer.ring_mla(
        tt_q,
        cache_kv,
        persistent_output_buffer_kv=buffer_kv,
        head_dim_v=GLOBAL_HEAD_DIM,
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
        is_balanced=False,
        slot_id=prefill_metadata.slot_idx,
        kv_actual_isl_tensor=prefill_metadata.kv_actual_global,
        kv_cache_num_layers=num_layers,
        kv_cache_layer_idx=layer_idx,
    )
    return out


# Whole-tile q chunks tried, smallest first, for unsplit global attention. q 160 overflows L1 beside k 256.
_GLOBAL_Q_CHUNKS = (96, 128)

# The slab whose global attention runs 4-tile Q chunks over five K-split bands. Its bf16 Q chunk leaves the ring SDPA's
# circular buffers 35 KB over L1; a bfp8 one is 61 KB smaller. Global Q@K^T runs at LoFi, which reads only the high
# mantissa bits anyway.
_BFP8_QUERY_SLAB_TOKENS = 256


def global_query_dtype(q_slab_tokens):
    """dtype the global ring SDPA takes Q in for this per-rank slab, or None to keep it."""
    return ttnn.bfloat8_b if q_slab_tokens == _BFP8_QUERY_SLAB_TOKENS else None


# What each chunk size gets at CP8 / TP4:
#                  q_chunk  k_chunk  K-split bands  segmented accumulation
#   global  2048       128      256              5  yes (bfp8 Q)
#   global  4096       128      256              3  yes
#   global  8192        96      256              1  yes
#   global 16384        96      256              1  no (too many Q chunks for the cores)
#   global 32768        96      256              1  no
#   sliding 2048       128      128              3  no
#   sliding 4096       128      128              3  no
#   sliding 8192+      128      128              1  no
def ring_sdpa_chunk_sizes(q_slab_tokens, sliding, num_heads=8, num_cores=110):
    """(q_chunk_size, k_chunk_size, max_k_splits, segmented_accumulation) for the ring SDPA, from the per-rank Q slab
    (chunk / CP), the local heads and the SDPA cores.
    """
    if sliding:
        # Split K three ways when the cores hold three bands of every (head, Q chunk) unit.
        q_chunk = int(os.environ.get("G4X_SLIDING_Q", 128))  # LOCAL EXPERIMENT knob
        k_splits = 3 if num_heads * -(-q_slab_tokens // q_chunk) * 3 <= num_cores else 1
        if os.environ.get("G4X_SLIDING_KSPLIT"):  # LOCAL EXPERIMENT
            k_splits = int(os.environ["G4X_SLIDING_KSPLIT"])
        return q_chunk, 128, k_splits, False
    # LOCAL EXPERIMENT: G4X_GLOBAL_QKS=q,k,splits[,seg] overrides the global choice.
    if os.environ.get("G4X_GLOBAL_QKS"):
        v = [int(x) for x in os.environ["G4X_GLOBAL_QKS"].split(",")]
        return v[0], v[1], v[2], bool(v[3]) if len(v) > 3 else True
    # A 256-row slab (chunk 2048): two 4-tile Q chunks per head are 16 units, two grid rows per band, so five
    # bands fill 80 cores. Per Q row a 4-tile chunk does about 1.3x the work of a 2-tile one in the same time, which
    # beats q64's three bands over 96 cores once the prefix is long. It needs the bfp8 Q of global_query_dtype.
    if q_slab_tokens == _BFP8_QUERY_SLAB_TOKENS:
        return 128, 256, 5, True
    # Short slabs: 4 Q chunks per head are too few to fill the grid, so K is split over 3 bands.
    if q_slab_tokens <= 512:
        q_chunk = q_slab_tokens // 4
        return (q_chunk if q_chunk % TILE_HEIGHT == 0 else TILE_HEIGHT), 256, 3, True
    # Segmented accumulation needs one Q chunk per core.
    for q_chunk in _GLOBAL_Q_CHUNKS:
        if -(-q_slab_tokens // q_chunk) * num_heads <= num_cores:
            return q_chunk, 256, 1, True
    return _GLOBAL_Q_CHUNKS[0], 256, 1, False


def ring_prefill_program_config(
    mesh_device,
    ccl_manager,
    head_dim,
    q_chunk_size,
    k_chunk_size,
    max_k_splits=1,
    matmul_math_fidelity=None,
    segmented_accumulation=False,
):
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
        max_k_splits=max_k_splits,
        matmul_math_fidelity=matmul_math_fidelity,
        segmented_accumulation=segmented_accumulation,
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
    gather_buffer_key,
    sliding_window_size=None,
    scale=1.0,
    compute_kernel_config=None,
    program_config=None,
    layer_idx=0,
    num_layers=1,
    slot_idx=0,
):
    """Attend this rank's Q shard over the cached prefix via the CP ring, with separate K and V caches.

    ``logical_n`` fixes the cache capacity at capture. Device metadata supplies
    the valid prefix on each replay; ``kv_actual_global`` is the prefix before
    this chunk when metadata is updated here.

    Returns ``[1, num_local_q_heads, q_local, head_dim]`` — this rank's rows only, so
    the output stays CP-sharded exactly like the input.
    """
    mesh_device = mesh_config.device
    if program_config is None:
        sdpa_grid = ccl_manager.compute_grid_size
        q_chunk, k_chunk, k_splits, _ = ring_sdpa_chunk_sizes(
            tt_q.shape[-2], sliding=True, num_heads=tt_q.shape[1], num_cores=(sdpa_grid.x - 1) * sdpa_grid.y
        )
        program_config = ring_prefill_program_config(
            mesh_device, ccl_manager, head_dim, q_chunk_size=q_chunk, k_chunk_size=k_chunk, max_k_splits=k_splits
        )
    # Only the predecessor halo is exchanged, and the op requires a compact buffer (gathered rows < cache_seq *
    # ring): a full-capacity one fails "requires a compact halo buffer". The halo is the window rounded up to whole
    # k chunks.
    k_chunk = program_config.k_chunk_size
    halo_tokens = -(-(sliding_window_size - 1) // k_chunk) * k_chunk
    gather_seq = max(halo_tokens, TILE_HEIGHT)
    halo_mc_k, halo_mc_v = cache_k.memory_config(), cache_v.memory_config()
    if os.environ.get("G4X_HALO_L1"):  # LOCAL EXPERIMENT: halo gather buffers in L1, two alternating pairs
        halo_mc_k = halo_mc_v = ttnn.L1_MEMORY_CONFIG
        gather_buffer_key = ("halo_l1", layer_idx % 2)
    buffer_k = ccl_manager.get_ring_gather_buffer(
        (gather_buffer_key, "ring_k"), num_local_kv_heads, gather_seq, head_dim, cache_k.dtype, halo_mc_k
    )
    buffer_v = ccl_manager.get_ring_gather_buffer(
        (gather_buffer_key, "ring_v"), num_local_kv_heads, gather_seq, head_dim, cache_v.dtype, halo_mc_v
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
        # LOCAL EXPERIMENT: G4X_SLIDING_RING=1 sends the halo over the CP ring's wrap link.
        topology=ttnn.Topology.Ring if os.environ.get("G4X_SLIDING_RING") else ttnn.Topology.Linear,
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
