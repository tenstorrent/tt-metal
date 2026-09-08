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
    slot_id_tensor=None,
    kv_actual_tensor=None,
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

    # --- trace-safe metadata path -------------------------------------------------------------
    # ring_joint and ring_mla are two front-ends over the same primitive, and that primitive reads
    # kv_cache_batch_idx (metadata[0]) and kv_actual_isl (metadata[1]) ON-DEVICE when both 1-element
    # uint32 tensors are supplied, deriving logical_nt / q-mapping / ring masks from them. That is
    # what lets ONE captured trace replay across chunks.
    #
    # Two things must change together, or the capture is silently wrong:
    #   * the host scalars must NOT be passed -- a supplied kv_actual_isl re-enables the host-side
    #     valid-pages patch, which a trace would freeze at chunk 0's value; and
    #   * logical_n must become a per-chunk CONSTANT. The cache's global capacity is the natural
    #     choice: it is identical for every chunk, so the capture carries no chunk-specific
    #     geometry at all and every replay derives its own from the metadata tensors.
    use_metadata = slot_id_tensor is not None and kv_actual_tensor is not None
    assert (slot_id_tensor is None) == (kv_actual_tensor is None), (
        "slot_id_tensor and kv_actual_tensor must be supplied together, or neither -- the op "
        "requires both to take the on-device path"
    )
    ring_logical_n = cache_global if use_metadata else logical_n

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
        logical_n=ring_logical_n,
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
        # On the metadata path the op does this fold itself from slot_id[0], num_layers and
        # layer_idx, so the host index is withheld and the tensors are passed instead.
        kv_cache_batch_idx=None if use_metadata else slot_idx * num_layers + layer_idx,
        kv_actual_isl=None if use_metadata else kv_actual,
        # Only forwarded on the metadata path: these kwargs exist on the ring_joint binding only
        # once the front-end plumbs them (the primitive has always accepted them). Passing them
        # unconditionally breaks the eager path on a build without that plumbing.
        **_metadata_kwargs(use_metadata, slot_id_tensor, kv_actual_tensor, num_layers, layer_idx),
    )
    return out


def _metadata_kwargs(use_metadata, slot_id_tensor, kv_actual_tensor, num_layers, layer_idx):
    """The trace-safe metadata kwargs, or nothing at all on the eager path.

    ``ttnn::prim::ring_joint_scaled_dot_product_attention`` -- the primitive shared by ring_joint and
    ring_mla -- has read slot_id / kv_actual_isl on-device for a while, but only the ``ring_mla``
    front-end plumbed the kwargs through its C++ wrapper and nanobind binding. On a build where
    ``ring_joint`` has not yet been given the same plumbing, naming these kwargs is a TypeError, so
    they are omitted entirely unless the metadata path is actually in use.
    """
    if not use_metadata:
        return {}
    return {
        "slot_id": slot_id_tensor,
        "kv_actual_isl_tensor": kv_actual_tensor,
        "kv_cache_num_layers": num_layers,
        "kv_cache_layer_idx": layer_idx,
    }
