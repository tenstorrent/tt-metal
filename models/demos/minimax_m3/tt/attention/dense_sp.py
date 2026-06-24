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


def _persistent_buf(mesh_device, rows, cols, n_kv, cache_global, head_dim):
    """Ring gather buffer: full cached prefix, dtype MUST match the (bf8) KV cache."""
    import torch

    return ttnn.from_torch(
        torch.zeros(1, n_kv, cache_global, head_dim),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=[None, 1]),
    )


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
):
    """Write this chunk's K/V into the chunked-KV cache, then ring_joint cache-read over the prefix.

    tt_q              [1, n_q_local, chunk_global, head_dim]   block-cyclic over the chunk, SP×TP sharded
    cache_k, cache_v  the chunked-KV caches (init_kvpe_cache), SP-sharded block-cyclic, bf8
    tt_k_chunk/v      [1, n_kv_local, chunk_global, head_dim]  this chunk, block-cyclic, to write
    kv_actual         prefix length already in the cache before this chunk (drives on-device rotation)
    logical_n         total valid prefix length (q attends causally over [0:logical_n])
    -> out            [1, n_q_local, chunk_local, head_dim]    block-cyclic over the chunk
    """
    rows, cols = tuple(mesh_device.shape)

    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache_k, tt_k_chunk, slot_idx=slot_idx, layer_idx=layer_idx, num_layers=num_layers,
        kv_actual_global=kv_actual, cluster_axis=cluster_axis,
    )
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache_v, tt_v_chunk, slot_idx=slot_idx, layer_idx=layer_idx, num_layers=num_layers,
        kv_actual_global=kv_actual, cluster_axis=cluster_axis,
    )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q, cache_k, cache_v,
        None, None, None,
        persistent_output_buffer_k=_persistent_buf(mesh_device, rows, cols, n_kv, cache_global, head_dim),
        persistent_output_buffer_v=_persistent_buf(mesh_device, rows, cols, n_kv, cache_global, head_dim),
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
        kv_cache_batch_idx=slot_idx,
        kv_actual_isl=kv_actual,
    )
    return out
