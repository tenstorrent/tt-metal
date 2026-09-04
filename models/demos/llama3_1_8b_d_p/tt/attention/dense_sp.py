# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B SP attention via the ring-joint SDPA over the block-cyclic KV cache.

Copied from ``gpt_oss_d_p/tt/attention/dense_sp.py``. ``ring_joint_scaled_dot_product_attention``
reads/gathers the KV across the SP axis *internally* via online softmax — there is no explicit
all-gather of K/V.

Two gpt-oss arguments are deliberately absent, not defaulted: ``attention_sink`` (Llama has no
learned sinks) and ``sliding_window_size`` (every Llama layer is full-causal). Their absence means
the ring op always takes its full-causal path and always gathers the whole per-device shard, so the
donor's compact-halo ``_gather_seq_len`` helper is dropped too.

The KV cache is already the DeepSeek chunked-KV substrate (same ``update_padded_kv_cache`` write,
same NdShard layout, same user-major packing ``slot = user*num_layers + layer`` ==
``kv_cache_batch_idx``), so no cache re-layout is needed. Grouped V (2 KV heads/chip at TP=4, no
inflation). ``is_balanced=False`` (chunked prefill).
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
    """Cache-read ring_joint over the accumulated prefix ``[0:logical_n]``.

    Args:
        tt_q: ``[1, n_q_local, chunk_local, head_dim]`` — block-cyclic over the chunk, SP x TP sharded
        cache_k, cache_v: the block-cyclic SP KV caches (``LlamaKVCache.k`` / ``.v``), bf8
        tt_k_chunk, tt_v_chunk: this chunk's K/V to write; ignored when ``write_chunk=False``
            (the per-layer seam already wrote them via ``write_kv_chunk``)
        kv_actual: valid prefix length already in the cache before this chunk (drives the
            on-device block-cyclic rotation)
        logical_n: total valid prefix length — q attends causally over ``[0:logical_n]``
        n_kv: GLOBAL KV head count (8). The ring gather buffer shards it across the TP cols, so it
            must be the global count, not the per-chip one.
        cache_global: total per-user cache capacity in tokens (``kv_cache.max_seq_len``)

    Returns:
        ``[1, n_q_local, chunk_local, head_dim]`` — block-cyclic over the chunk.
    """
    assert cache_k.dtype == ttnn.bfloat8_b and cache_v.dtype == ttnn.bfloat8_b, (
        f"chunked ring cache-read requires a bf8 KV cache; got k={cache_k.dtype}, v={cache_v.dtype}. "
        "The ring path and its gather buffers are bf8."
    )
    if write_chunk:
        for cache, chunk in ((cache_k, tt_k_chunk), (cache_v, tt_v_chunk)):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                chunk,
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
        # Persistent ring-gather scratch, allocated once per (key, size) and reused across every
        # layer and chunk. Full-causal layers gather the entire per-device shard. dtype MUST match
        # the bf8 KV cache.
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            f"dense_k_{cache_global}", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            f"dense_v_{cache_global}", n_kv, cache_global, head_dim, ttnn.bfloat8_b
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
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        # Fold the layer into the cache batch index, matching update_padded_kv_cache's write
        # (batch_idx = slot*num_layers + layer). Passing slot alone makes every layer read layer 0's
        # cache: L0 correct by coincidence, L1+ read stale and corrupt attention.
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out
