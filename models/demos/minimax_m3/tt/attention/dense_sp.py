# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 dense attention under SP — the real model forward for the dense layers (0-2).

Chunked-prefill GQA attention via the chunked-KV cache + ring_joint SDPA. Unlike MSA, ring_joint
reads the KV cache and gathers the accumulated prefix across the SP axis *internally* (online-softmax
over the ring), so there is no explicit AllGather here:

  update_padded_kv_cache(cache, this-chunk K/V)            write the chunk into the SP-sharded cache
  ring_joint_scaled_dot_product_attention(q, cache_k, cache_v, kv_actual_isl, logical_n)
                                                            causal GQA over the cached prefix [0:logical_n]

Grouped V (cache stays n_kv heads, 1/chip at TP=4 — NO inflation; Pavle's GQA-causal kernel). No
balancing / zigzag for chunked prefill (is_balanced=False). The validated building block is
tests/unit/test_ring_joint_cache_read_sp_vs_ref.py (PCC 0.99994); this is that mechanism as a callable
model forward. Perf config q_chunk=128 / k_chunk=512 (Pavle's minimax3_gqa_causal_perf).
"""

import ttnn


def dense_sp_attention(
    tt_q,
    cache_k,
    cache_v,
    tt_k_chunk,
    tt_v_chunk,
    *,
    kv_actual,
    logical_n,
    n_kv,
    cache_global,
    head_dim,
    mesh_device,
    ccl_manager,
    program_config,
    compute_kernel_config,
    scale,
    cluster_axis,
    slot_idx=0,
    layer_idx=0,
    num_layers=1,
    write_chunk=True,
):
    """Write this chunk's K/V into the chunked-KV cache (unless already written), then ring_joint
    cache-read over the prefix.

    tt_q              [1, n_q_local, chunk_global, head_dim]   block-cyclic over the chunk, SP×TP sharded
    cache_k, cache_v  the chunked-KV caches (init_kvpe_cache), SP-sharded block-cyclic, bf8
    tt_k_chunk/v      [1, n_kv_local, chunk_global, head_dim]  this chunk, block-cyclic, to write
                      (ignored when write_chunk=False — e.g. the model seam already wrote it)
    kv_actual         prefix length already in the cache before this chunk (drives on-device rotation)
    logical_n         total valid prefix length (q attends causally over [0:logical_n])
    write_chunk       when False, skip the cache write and only read (the per-layer seam is the writer)
    -> out            [1, n_q_local, chunk_local, head_dim]    block-cyclic over the chunk
    """
    if write_chunk:
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache_k,
            tt_k_chunk,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual,
            cluster_axis=cluster_axis,
        )
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache_v,
            tt_v_chunk,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual,
            cluster_axis=cluster_axis,
        )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch, allocated once by the CCL manager and reused across all
        # layers/chunks (was a per-call from_torch(zeros)). dtype MUST match the (bf8) KV cache.
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "dense_k", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "dense_v", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        # Fold the layer into the cache batch index, matching update_padded_kv_cache's write
        # (batch_idx = slot_idx*num_layers + layer_idx). The cache packs all layers user-major in the
        # batch dim; passing slot_idx alone made every dense layer read layer 0's cache (L0 correct by
        # coincidence, L1+ read stale L0 K/V -> wrong attn_out -> residual corruption -> KV-PCC crater).
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out


def dense_sp_attention_nocache(
    tt_q,
    tt_k,
    tt_v,
    *,
    mesh_config,
    ccl_manager,
    logical_n,
    n_kv,
    head_dim,
    scale,
    program_config,
    compute_kernel_config,
):
    """First-chunk dense SP attention: ring_joint over the chunk's OWN SP-sharded K/V (NO persistent cache).

    Each device's query shard attends to the full `logical_n` sequence reconstructed across the SP ring
    (grouped V, no inflation, is_balanced=False). For the first prefill chunk where there's no prior
    cache; multi-chunk accumulation uses dense_sp_attention (cache-read). Validated op-level by
    tests/unit/test_ring_joint_sp_vs_ref.py (PCC 0.99998). Returns the per-device query-shard output.

    n_kv is the GLOBAL KV-head count (e.g. 4); the ring-gather persistent buffer shards it across the TP
    cols (1/device at TP=4), matching the per-device KV head that tt_k/tt_v already carry.
    """
    mesh_device = ccl_manager.mesh_device
    sp_axis = mesh_config.sp_axis

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch (heads on TP cols, seq replicated across SP -> dims=[None, 1]),
        # allocated once by the CCL manager and reused across chunks. First-chunk K/V are bf16.
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "nocache_k", n_kv, logical_n, head_dim, ttnn.bfloat16
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "nocache_v", n_kv, logical_n, head_dim, ttnn.bfloat16
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
    )
    return out
