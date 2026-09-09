# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B GQA attention under sequence parallelism — the ring SDPA read path.

This is the **structural** half of attention (recipe §2.3): it encodes how the KV cache is read, so
it is written in D2 alongside the cache layout rather than left to D3. It is a near-verbatim port of
`minimax_m3/tt/attention/dense_sp.py` (its dense layers 0-2), which was measured at head_dim 128,
chunk 5120, sp8×tp4 on this same Blackhole Galaxy — the identical envelope for every dimension the
ring op cares about.

`ring_joint_scaled_dot_product_attention` reads the KV cache and gathers the accumulated prefix
across the SP axis **internally**, by online softmax over the ring. There is therefore no explicit
AllGather here, and no `repeat_kv`-style head inflation: the kernel is GQA-causal and consumes
grouped V directly, so the cache stays at `n_kv` heads.

Two entry points, both the same op:

* :func:`dense_sp_attention_nocache` — first chunk, no prior cache: ring over the chunk's OWN
  SP-sharded K/V.
* :func:`dense_sp_attention` — cache-read: ring over the prefix accumulated in the block-cyclic
  cache. This is the mechanism chunked prefill depends on.

## Two heads per chip

`n_kv` here is the **GLOBAL** KV head count (8). The persistent ring-gather buffer shards it across
the TP cols, giving **2 heads per chip** at TP=4 — not the 1 that every donor package has. The
buffer allocation and the per-device K/V it is reconstructing must agree on that, and nothing raises
if they do not. See `kv_cache.py` for the full note.

## `kv_cache_batch_idx`

The cache packs all layers user-major in the batch dim, so the read index must be
`slot_idx * num_layers + layer_idx` — the same arithmetic `update_padded_kv_cache` writes at.
Passing `slot_idx` alone makes every layer read layer 0's cache: layer 0 is then correct by
coincidence and every later layer reads stale K/V, which corrupts the residual and craters the KV
PCC. The donor hit exactly this; it is called out here so the port cannot repeat it.
"""

import ttnn


def _ring_sdpa(
    tt_q,
    k_source,
    v_source,
    *,
    ccl_manager,
    mesh_device,
    cluster_axis,
    logical_n,
    n_kv,
    buffer_seq,
    buffer_dtype,
    buffer_prefix,
    head_dim,
    scale,
    program_config,
    compute_kernel_config,
    kv_cache_batch_idx=None,
    kv_actual_isl=None,
):
    """The single `ring_joint_scaled_dot_product_attention` call both paths share.

    The persistent ring-gather scratch is allocated once by the CCL manager and reused across every
    layer and chunk (a per-call `from_torch(zeros)` churns host and DRAM on every attention). The op
    treats it as pure scratch — it fills the gathered region and masks the invalid tail via
    `kv_actual_isl` — so reuse without re-zeroing is safe. Its dtype MUST match the K/V it gathers:
    bf8 on the cache-read path (the cache dtype), bf16 on the first-chunk path (live activations).
    """
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        k_source,
        v_source,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            f"{buffer_prefix}_k", n_kv, buffer_seq, head_dim, buffer_dtype
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            f"{buffer_prefix}_v", n_kv, buffer_seq, head_dim, buffer_dtype
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
        is_balanced=False,  # no zigzag balancing for chunked prefill
        **({} if kv_cache_batch_idx is None else {"kv_cache_batch_idx": kv_cache_batch_idx}),
        **({} if kv_actual_isl is None else {"kv_actual_isl": kv_actual_isl}),
    )
    return out


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
    cache_dtype=ttnn.bfloat8_b,
):
    """Cache-read path: optionally write this chunk's K/V, then ring over the cached prefix.

    ```
    tt_q              [1, n_q_local,  chunk_global, head_dim]  block-cyclic, SP x TP sharded
    cache_k, cache_v  the packed caches, SP-sharded block-cyclic, bf8
    tt_k_chunk/v      [1, n_kv_local, chunk_global, head_dim]  this chunk, to write
                      (ignored when write_chunk=False - the per-layer seam is then the writer)
    kv_actual         prefix length already in the cache before this chunk (drives the on-device
                      block-cyclic rotation)
    logical_n         total valid prefix length; q attends causally over [0:logical_n]
    cache_global      FULL cache capacity in tokens - the ring-gather buffer must span all of it,
                      because the op gathers the entire per-device cache shard
    -> out            [1, n_q_local, chunk_local, head_dim]
    ```
    """
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

    return _ring_sdpa(
        tt_q,
        cache_k,
        cache_v,
        ccl_manager=ccl_manager,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        logical_n=logical_n,
        n_kv=n_kv,
        buffer_seq=cache_global,
        buffer_dtype=cache_dtype,
        buffer_prefix="dense",
        head_dim=head_dim,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        # Fold the layer into the cache batch index, matching the write. See the module docstring.
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )


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
    """First-chunk path: ring over the chunk's OWN SP-sharded K/V, with no persistent cache.

    Each device's query shard attends the full `logical_n` sequence reconstructed across the SP
    ring. Used for a one-shot prefill and for chunk 0 of a chunked run when the cache is empty;
    multi-chunk accumulation goes through :func:`dense_sp_attention`. Returns the per-device query
    shard's output.
    """
    return _ring_sdpa(
        tt_q,
        tt_k,
        tt_v,
        ccl_manager=ccl_manager,
        mesh_device=ccl_manager.mesh_device,
        cluster_axis=mesh_config.sp_axis,
        logical_n=logical_n,
        n_kv=n_kv,
        buffer_seq=logical_n,
        buffer_dtype=ttnn.bfloat16,  # first-chunk K/V are live bf16 activations, not the bf8 cache
        buffer_prefix="nocache",
        head_dim=head_dim,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
