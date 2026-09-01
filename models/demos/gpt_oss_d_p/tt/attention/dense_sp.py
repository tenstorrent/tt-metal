# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS SP attention via Pavle's ring-joint SDPA — replaces the gather-Q stand-in.

Ported from ``minimax_m3/tt/attention/dense_sp.py``. The cache-read path uses
``ttnn.transformer.ring_joint_scaled_dot_product_attention`` (the ring reads/gathers the KV across the
SP axis *internally* via online-softmax — no explicit AllGather):

  * :func:`dense_sp_attention` — cache-backed ring_joint over the block-cyclic SP KV cache for
    every chunk, including chunk 0. The fixed/growing cache makes Q shorter than K/V even for
    the first complete Q group, allowing one ring program across the entire prefill.

GPT-OSS vs M3: gpt-oss attention has **attention sinks** and **per-layer sliding-window** masking, so
both paths thread ``attention_sink`` + ``sliding_window_size`` into the op (M3 has neither). These are
Pavle's ``ring-joint-sdpa-attention-sinks`` additions, merged in #51438. ``attention_sink`` is our ``weights.sinks`` (stored ``sinks_div_scale`` = sink×√d, the
convention the op expects); ``sliding_window_size`` is the per-layer window (None on full layers).

Our GptOssKVCache is already the DeepSeek chunked-KV substrate M3 uses (same update_padded_kv_cache
write, same NdShard layout, same user-major packing slot=user*num_layers+layer == kv_cache_batch_idx),
so no cache re-layout is needed. Grouped V (n_kv heads, 1/chip at TP=8; no inflation). is_balanced=False
(chunked prefill). Ring building block validated in M3 by test_ring_joint_{sp,cache_read}_vs_ref.py.
"""

import math

import ttnn


def _gather_seq_len(sliding_window_size, k_chunk_size, full_seq):
    """Ring gather-buffer seq length. Pavle's GPT-OSS sliding op requires a COMPACT halo buffer (not the
    full sequence) — halo = ceil((window-1)/k_chunk)*k_chunk, floored at one tile (per his test). Full
    layers (sliding_window_size=None) gather the whole sequence. Buffers are keyed by this length so a
    sliding layer (compact) and a full layer (full) get distinct CCL-manager buffers."""
    if sliding_window_size is None:
        return full_seq
    halo = math.ceil((sliding_window_size - 1) / k_chunk_size) * k_chunk_size
    return max(halo, ttnn.TILE_SIZE)


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
    attention_sink=None,
    sliding_window_size=None,
    bounded_kv_slab_count=None,
    slot_idx=0,
    layer_idx=0,
    num_layers=1,
    write_chunk=True,
):
    """Cache-read ring_joint over the accumulated prefix [0:logical_n] (multi-chunk / chunked prefill).

    tt_q              [1, n_q_local, chunk_global, head_dim]  block-cyclic over the chunk, SP×TP sharded
    cache_k, cache_v  the block-cyclic SP KV caches (GptOssKVCache.k/.v), bf8
    tt_k_chunk/v      this chunk's K/V to write (ignored when write_chunk=False — the per-layer seam
                      already wrote it via write_kv_chunk)
    kv_actual         valid prefix length already in the cache before this chunk (drives on-device rotation)
    logical_n         total valid prefix length (q attends causally over [0:logical_n])
    attention_sink    per-query-head sink (weights.sinks, bf16); sliding_window_size None on full layers
    bounded_kv_slab_count  bounded circular sliding cache (PR2): the cache holds only this many chunk
                      slabs and the ring read wraps its local slab addressing (chunk group g in slab
                      g mod n_slabs). None on full layers / unbounded caches. kv_actual and logical_n
                      stay TRUE ABSOLUTE lengths either way — the op derives the wrap on-device.
    -> out            [1, n_q_local, chunk_local, head_dim]  block-cyclic over the chunk
    """
    assert bounded_kv_slab_count is None or sliding_window_size is not None, (
        "bounded_kv_slab_count is only meaningful for sliding-window layers "
        f"(got bounded_kv_slab_count={bounded_kv_slab_count}, sliding_window_size=None)"
    )
    assert cache_k.dtype == ttnn.bfloat8_b and cache_v.dtype == ttnn.bfloat8_b, (
        f"chunked ring cache-read requires a bf8 KV cache; got k={cache_k.dtype}, v={cache_v.dtype}. "
        "KV_CACHE_DTYPE=bf16 is not supported for chunked prefill (the sliding RingJointSDPA path "
        "and its gather buffers are bf8)."
    )
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

    # Ring gather-buffer seq: full cache for full-attn layers; compact halo for sliding layers.
    _bufseq = _gather_seq_len(sliding_window_size, program_config.k_chunk_size, cache_global)
    # Full-attn layers pass sliding_window_size=None -> non-sliding full-causal ring path (sinks now
    # allowed after the device-op assert relax). Sliding layers pass 128. Buffer handles None (full).
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch (once per key/size, reused across layers/chunks). Full-attn
        # layers gather the entire per-device shard (cache_global); sliding layers use a compact halo
        # (_bufseq). logical_n / kv_actual_isl drive causal masking. dtype MUST match the bf8 KV cache.
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            f"dense_k_{_bufseq}", n_kv, _bufseq, head_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            f"dense_v_{_bufseq}", n_kv, _bufseq, head_dim, ttnn.bfloat8_b
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
        # Plumbed from the runtime config: Ring on torus pods, Linear elsewhere (op supports both).
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        # Fold the layer into the cache batch index, matching update_padded_kv_cache's write
        # (batch_idx = slot*num_layers + layer). Passing slot alone makes every layer read layer 0's
        # cache (L0 correct by coincidence, L1+ read stale -> attn corruption).
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
        # GPT-OSS additions (Pavle's sinks+sliding branch):
        attention_sink=attention_sink,
        sliding_window_size=sliding_window_size,
        # Bounded circular sliding KV cache (PR2). None => unbounded, byte-identical behavior.
        bounded_kv_slab_count=bounded_kv_slab_count,
    )
    return out
