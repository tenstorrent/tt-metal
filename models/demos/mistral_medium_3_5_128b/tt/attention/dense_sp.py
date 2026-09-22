# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The two sequence-parallel attention cores. Ported from ``gpt_oss_d_p/tt/attention/dense_sp.py``.

With the sequence SP-sharded across the mesh rows, every chip holds only ``tokens/sp`` of the
queries *and* only ``tokens/sp`` of the keys, but causal attention needs each query to see every
earlier key. There are two ways to close that gap, and which is cheaper depends on how much KV
there is relative to Q:

``gathered_sp_attention``
    Run the same ring op over the chunk's **own** live SP-sharded K/V, with no cache behind it.
    Right when Q and KV are the same length (a one-shot prefill, no prior cache).

    The D2 interface sketch had this doing an explicit all-gather of K/V, a plain causal SDPA, and
    a reduce-scatter back with a ``1/sp`` rescale. It does not: the ring op already gathers K/V
    across the SP axis internally with an online softmax, so the explicit version would be a
    second, unvalidated mechanism that materialises the whole KV per chip for no gain.
    ``tests/unit/test_ring_joint_sp_vs_ref.py::test_ring_joint_sp_live_qkv_vs_ref`` measures this
    exact op parameterisation at s2048 and s5120, which is why it is the one kept. The name is
    still accurate — the gather happens, just inside the op.

``dense_sp_attention``
    Leave K/V in the block-cyclic DRAM cache and run
    ``ttnn.transformer.ring_joint_scaled_dot_product_attention``, which walks the ring with an
    online softmax so no chip ever materialises the full KV. Right when the cache is long relative
    to the chunk — i.e. every chunk after the first, and any run whose cache capacity exceeds the
    current sequence. ``is_balanced=False`` because the causal work per chip is triangular, not
    uniform; ``use_column_major_ccl=True`` matches the SP-on-rows layout; ``joint_strategy="rear"``
    appends the live chunk's own K/V after the cached prefix.

Both paths finish with heads still split; ``prefill.attention_forward`` does the concat and o_proj.
"""

import ttnn


def dense_sp_attention(
    tt_q,
    cache_k,
    cache_v,
    tt_k,
    tt_v,
    *,
    kv_actual: int,
    logical_n: int,
    n_kv: int,
    cache_global: int,
    head_dim: int,
    mesh_device,
    ccl_manager,
    program_config,
    compute_kernel_config,
    scale: float,
    cluster_axis: int,
    slot_idx: int = 0,
    layer_idx: int = 0,
    num_layers: int = 1,
    write_chunk: bool = True,
):
    """Write this chunk's K/V into the cache, then ring-joint SDPA over the whole prefix.

    Argument shape follows ``minimax_m3/tt/attention/dense_sp.py``, which the validated cache-read
    tests are written against — with the trace metadata-tensor forms (``slot_id`` /
    ``kv_actual_isl_tensor``) left out. Those exist to let one captured trace re-target its user and
    depth between replays, which belongs to serving; this bring-up dispatches eagerly and passes
    the slot as a host integer.

    Args:
        tt_q: ``[1, n_heads_local, q_local, head_dim]`` — this chip's queries, post-RoPE.
        cache_k, cache_v: the cache tensors from ``kv_cache.allocate_kv_cache``.
        tt_k, tt_v: this chunk's ``[1, n_kv, chunk_local, head_dim]`` K/V, post-RoPE for K. Written
            into the cache through ``kv_cache.write_kv_chunk`` when ``write_chunk``; may be None
            when the caller has already written them.
        kv_actual: tokens already **written** to the cache for this user/layer, i.e. the live
            chunk's absolute start. A multiple of ``chunk_size``, which is what makes the
            block-cyclic rotation degenerate to a contiguous SP split.
        logical_n: total logical KV length the op should attend over (``kv_actual`` + this chunk).
        n_kv: KV heads per chip (2 at target).
        cache_global: the cache's global token capacity (``round_cache_capacity`` output).
        head_dim: 128.
        scale: softmax scale, ``1/sqrt(head_dim)``.
        cluster_axis: the SP mesh axis (0 here — SP on rows).
        slot_idx: the cache's user slot; folded with ``layer_idx`` as
            ``slot_idx * num_layers + layer_idx``.
        layer_idx, num_layers: the layer's position in the fold.
        write_chunk: False to read a prefix that is already fully written.

    Returns:
        ``[1, n_heads_local, q_local, head_dim]``."""
    if write_chunk:
        assert tt_k is not None and tt_v is not None, "write_chunk=True needs this chunk's K and V"
        for cache, chunk in ((cache_k, tt_k), (cache_v, tt_v)):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                chunk,
                slot_idx=slot_idx,
                kv_actual_global=kv_actual,
                layer_idx=layer_idx,
                num_layers=num_layers,
                cluster_axis=cluster_axis,
            )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch, allocated once by the CCL manager and reused across every
        # layer and chunk. The dtype MUST match the cache's (bfloat8_b).
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
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        kv_cache_batch_idx=slot_idx,
        kv_actual_isl=kv_actual,
        kv_cache_num_layers=num_layers,
        kv_cache_layer_idx=layer_idx,
    )
    return out


def gathered_sp_attention(
    tt_q,
    tt_k,
    tt_v,
    *,
    mesh_config,
    ccl_manager,
    program_config,
    compute_kernel_config,
    mesh_device,
    scale: float,
    seq_len_global: int,
):
    """One-shot SP attention: the ring op over the chunk's own live K/V, no cache.

    Args:
        tt_q, tt_k, tt_v: ``[1, heads_local, tokens_local, head_dim]``, SP-sharded on tokens.
            K is post-RoPE; V is raw.
        seq_len_global: ``tokens_local * sp``; the causal length the op attends over.
        compute_kernel_config: added over the D2 sketch — the ring op takes one, and it must have
            ``fp32_dest_acc_en=False`` (see :meth:`ProgramConfig.get_ring_compute_kernel_config`).

    Returns:
        ``[1, n_heads_local, tokens_local, head_dim]``, SP-sharded again.
    """
    head_dim = tt_k.shape[-1]
    # The persistent buffer is keyed by the GLOBAL KV-head count; the mapper shards it back over TP.
    n_kv_global = tt_k.shape[1] * mesh_config.tp
    dtype = tt_k.dtype

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "oneshot_k", n_kv_global, seq_len_global, head_dim, dtype
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "oneshot_v", n_kv_global, seq_len_global, head_dim, dtype
        ),
        joint_strategy="rear",
        logical_n=seq_len_global,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=mesh_config.sp_axis,
        mesh_device=mesh_device,
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
    )
    return out
