# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-parallel GQA attention via ``ring_joint_scaled_dot_product_attention``.

Under SP=8 each device holds ``s_local = chunk / 8`` query rows and the matching K/V shard. The ring
op gathers the other rows' K/V across the SP axis *internally* by online softmax, so there is no
explicit all-gather here and no host-side mask: causality is derived from ``logical_n`` and (on the
cache path) ``kv_actual_isl``.

Two entry points, one op:

* :func:`ring_sdpa_live` — first chunk, no accumulated prefix: the ring reads the chunk's OWN
  SP-sharded K/V.
* :func:`ring_sdpa_cache_read` — chunk N > 0: the ring reads the block-cyclic KV cache over the
  accumulated prefix ``[0, logical_n)``.

Two things here are load-bearing and were wrong-by-default in the source until they were fixed
there; both are asserted or commented so they cannot regress:

* ``kv_cache_batch_idx`` must be ``slot_idx * num_layers + layer_idx``, not ``slot_idx``. The cache
  packs all layers user-major in the batch dim, so passing the slot alone makes every layer read
  layer 0's K/V — layer 0 correct by coincidence, every other layer reading stale K/V.
* the ring-gather scratch must span the FULL cache capacity (``max_seq_len``), not ``logical_n``.
  The op gathers each device's entire cache shard regardless of how much of it is valid; sizing the
  buffer to ``logical_n`` happens to work for a 2-chunk run (where the last chunk's ``logical_n``
  equals the capacity) and fails on the third chunk.

GQA is grouped natively (``NKH == NVH < NQH``, ``NQH % NKH == 0``): at TP=4 each chip carries 8 query
heads over 2 KV heads and the cache is NOT inflated to 8.
"""

from __future__ import annotations

import ttnn


def ring_sdpa_live(
    tt_q,
    tt_k,
    tt_v,
    *,
    mesh_config,
    ccl_manager,
    logical_n: int,
    n_kv_global: int,
    head_dim: int,
    scale: float,
    program_config,
    compute_kernel_config,
):
    """First-chunk SP attention over the chunk's own K/V. Returns the per-device query-shard output.

    ``n_kv_global`` is the GLOBAL KV-head count; the persistent ring-gather buffer shards it across
    the TP cols so each device sees the heads it actually holds.
    """
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "live_k", n_kv_global, logical_n, head_dim, tt_k.dtype
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "live_v", n_kv_global, logical_n, head_dim, tt_v.dtype
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=mesh_config.sp_axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
    )
    return out


def ring_sdpa_cache_read(
    tt_q,
    cache_k,
    cache_v,
    *,
    mesh_config,
    ccl_manager,
    kv_actual: int,
    logical_n: int,
    cache_global: int,
    n_kv_global: int,
    head_dim: int,
    scale: float,
    program_config,
    compute_kernel_config,
    slot_idx: int,
    layer_idx: int,
    num_layers: int,
):
    """Chunk N > 0: this chunk's Q against the accumulated prefix already in the cache.

    ``kv_actual`` is the valid prefix BEFORE this chunk (which the caller has already written), and
    ``logical_n`` is the total valid prefix including it.
    """
    assert logical_n > kv_actual, f"logical_n {logical_n} must exceed the prior prefix {kv_actual}"
    assert logical_n <= cache_global, f"logical_n {logical_n} exceeds cache capacity {cache_global}"
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # dtype MUST match the (bf8) cache; the scratch spans the whole capacity, not logical_n.
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "cache_k", n_kv_global, cache_global, head_dim, cache_k.dtype
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "cache_v", n_kv_global, cache_global, head_dim, cache_v.dtype
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=mesh_config.sp_axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        # Fold the layer into the cache batch index, matching update_padded_kv_cache's write.
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out
