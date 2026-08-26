# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 sequence-parallel attention via the ring-joint SDPA.

Ported from ``gpt_oss_d_p/tt/attention/dense_sp.py`` (itself from ``minimax_m3``), minus the two
gpt-oss-only features: **no attention sinks** and **no sliding window** — Mistral is dense causal on
every layer, so the ring always gathers the full per-device cache shard and the compact-halo buffer
sizing is not needed.

``ttnn.transformer.ring_joint_scaled_dot_product_attention`` reads/gathers the KV across the SP axis
internally via online softmax, so there is no explicit AllGather of K/V.
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

    tt_q              [1, n_q_local, chunk_local, head_dim]  block-cyclic over the chunk, SPxTP sharded
    cache_k, cache_v  the block-cyclic SP KV caches (MistralKVCache.k/.v), bf8
    tt_k_chunk/v      this chunk's K/V to write (ignored when write_chunk=False — the per-layer seam
                      already wrote it via write_kv_chunk)
    kv_actual         valid prefix already in the cache before this chunk (drives on-device rotation)
    logical_n         total valid prefix length (q attends causally over [0:logical_n])
    -> out            [1, n_q_local, chunk_local, head_dim]
    """
    assert (
        cache_k.dtype == ttnn.bfloat8_b and cache_v.dtype == ttnn.bfloat8_b
    ), f"chunked ring cache-read requires a bf8 KV cache; got k={cache_k.dtype}, v={cache_v.dtype}."
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

    # Dense causal on every layer => always gather the whole per-device cache shard.
    # `n_kv` here is the FULL KV head count (8); get_ring_gather_buffer shards zeros(1, n_kv, ...)
    # across the TP cols, so each chip's buffer holds n_kv/tp = 2 heads at TP=4 — matching the cache
    # and the per-chip Q slice (24 heads). The op validates grouped GQA as NKH == NVH < NQH with
    # NQH % NKH == 0, i.e. 2/2/24 here.
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch, allocated once per (key, shape, dtype) and reused across
        # every layer/chunk. The op fills the gathered region and masks the invalid tail, so reuse
        # without re-zeroing is safe. dtype MUST match the bf8 KV cache.
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
        # cache — layer 0 correct by coincidence, every later layer silently stale.
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out
