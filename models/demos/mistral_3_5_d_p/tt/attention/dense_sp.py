# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 SP attention via the ring-joint SDPA — the cluster's ``attention_read_path``.

Adapted from ``gpt_oss_d_p/tt/attention/dense_sp.py`` (which is itself M3's). The cache-read path
uses ``ttnn.transformer.ring_joint_scaled_dot_product_attention``: the ring reads/gathers the KV
across the SP axis *internally* by online softmax, so there is no explicit AllGather of K/V and
peak live memory stays bounded by the ring scratch rather than the whole cache.

:func:`dense_sp_attention` is cache-backed for EVERY chunk, chunk 0 included. The fixed (growing)
cache makes Q shorter than K/V even for the first complete Q group, which lets one ring program
cover the entire prefill.

Adapted from the donor:
  * **attention sinks removed** — no ``attention_sink`` argument to thread; Ministral3 has none.
  * **sliding window removed** — ``config.sliding_window`` is null for every Mistral layer, so every
    layer is full-causal. That collapses the donor's ``_gather_seq_len`` (which sized a COMPACT halo
    buffer for sliding layers and the full sequence for full ones) down to the full-sequence branch,
    and with it the dual buffer keying: every layer now shares one pair of ring-gather buffers.
  * **head_dim 64 -> 128**; ``num_kv_heads`` stays 8, so the 1-KV-head-per-TP-column layout, the
    32-token DRAM bank walk and ROUND_ROBIN_1D all carry over unchanged.

Our :class:`~.kv_cache.MistralKVCache` is the same DeepSeek chunked-KV substrate (same
``update_padded_kv_cache`` write, same NdShard layout, same user-major packing
``slot = user*num_layers + layer == kv_cache_batch_idx``), so no cache re-layout is needed. Grouped V
(n_kv heads, 1/chip at TP=8; no inflation). ``is_balanced=False`` (chunked prefill).
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

    tt_q              [1, n_q_local, chunk_global, head_dim]  block-cyclic over the chunk, SP x TP sharded
    cache_k, cache_v  the block-cyclic SP KV caches (MistralKVCache.k/.v), bf8
    tt_k_chunk/v      this chunk's K/V to write (ignored when ``write_chunk=False`` — the per-layer
                      seam already wrote it via ``write_kv_chunk``)
    kv_actual         valid prefix length already in the cache before this chunk (drives the
                      on-device block-cyclic rotation)
    logical_n         total valid prefix length (q attends causally over ``[0:logical_n]``)
    cache_global      the cache's allocated capacity in tokens (``spec.cache_capacity``)
    -> out            [1, n_q_local, chunk_local, head_dim]  block-cyclic over the chunk
    """
    # Fail-loud guard kept from the donor: the ring path and its gather buffers are bf8. A bf16 cache
    # would not raise inside the op, it would silently mis-read.
    assert cache_k.dtype == ttnn.bfloat8_b and cache_v.dtype == ttnn.bfloat8_b, (
        f"chunked ring cache-read requires a bf8 KV cache; got k={cache_k.dtype}, v={cache_v.dtype}. "
        "The spec fixes dataformats.kv_cache.default = bfloat8_b; a bf16 cache is not supported here."
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

    # Every Mistral layer is full-causal, so the ring always gathers the entire per-device shard —
    # there is no sliding halo, and therefore exactly one buffer size across the whole model.
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch (allocated once per key/size in the CCL manager, reused
        # across every layer and chunk). dtype MUST match the bf8 KV cache.
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
        # Ring on torus pods, Linear elsewhere (the op supports both); plumbed from the runtime.
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        # Fold the layer into the cache batch index, matching update_padded_kv_cache's write
        # (batch_idx = slot*num_layers + layer). Passing slot alone makes every layer read layer 0's
        # cache (L0 correct by coincidence, L1+ read stale -> attention corruption).
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out
