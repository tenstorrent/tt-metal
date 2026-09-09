# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention forward for prefill.

Borrowed from `minimax_m3/tt/attention/prefill.py`, minus its QK-norm and its MSA index branch.

    x (full emb)
      -> q_proj / k_proj / v_proj          column-parallel; 8 q heads and 2 KV heads per chip
      -> head split                        [1, n_local_heads, tokens_local, head_dim]
      -> RoPE on Q and K                   full-width rotation, Meta-format tables
      -> write K/V into the packed cache   update_padded_kv_cache, block-cyclic on the SP axis
      -> ring SDPA                         causal GQA over the cached prefix (dense_sp.py)
      -> o_proj                            row-parallel
      -> TP reduce-scatter / all-reduce    per the residual scheme

There is no QK-norm step and no attention-sink term: both are donor features Llama lacks.

**K is cached after RoPE; V is cached raw.** That is what the reference stores and what the golden
trace grades against — caching K pre-RoPE would force a re-rotation on every cache read.
"""

import ttnn

from .dense_sp import dense_sp_attention, dense_sp_attention_nocache
from .kv_cache import write_kv_chunk
from .operations import apply_reduce_scatter, apply_rope, split_qkv_heads


def attention_forward(
    x,
    *,
    weights,
    config,
    program_config,
    mesh_config,
    ccl_manager,
    mesh_device,
    rope_mats,
    transformation_mats,
    kv_cache=None,
    slot_idx=0,
    layer_idx=0,
    cached_len=0,
    logical_n=None,
    indexed_rope=False,
    write_chunk=True,
):
    """Run prefill attention for one chunk.

    Args:
        x: `[1, 1, tokens_local, hidden]` — full emb (the layer gathered it for the column-parallel
            projections).
        rope_mats: `(cos, sin)` in META order. Per-chunk tables when `indexed_rope` is False; the
            whole-cache block-cyclic SP-sharded tables when it is True.
        kv_cache: :class:`~.kv_cache.LlamaKVCache`, or None to run without a cache (the one-shot
            path the isolated attention test uses).
        cached_len: valid prefix length already in the cache, BEFORE this chunk. 0 for one-shot.
        logical_n: total valid prefix the queries attend over. Defaults to `cached_len + tokens`.
        write_chunk: False when the per-layer seam already wrote this chunk's K/V.

    Returns the attention output, sharded per the residual scheme.
    """
    tokens_local = x.shape[-2]
    sp = mesh_config.sp
    tokens_global = tokens_local * sp
    if logical_n is None:
        logical_n = cached_len + tokens_global

    compute_cfg = program_config.get_compute_kernel_config()
    n_q_local = config.q_heads_per_chip(mesh_config.tp)
    n_kv_local = config.kv_heads_per_chip(mesh_config.tp)

    # --- projections. Column-parallel: full-emb input, per-chip head slice out. ---
    q = ttnn.linear(x, weights.q_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_cfg)
    k = ttnn.linear(x, weights.k_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_cfg)
    v = ttnn.linear(x, weights.v_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_cfg)

    # --- head split: [1, 1, tokens, n_local*hd] -> [1, n_local, tokens, hd] ---
    q = split_qkv_heads(q, num_local_heads=n_q_local, head_dim=config.head_dim)
    k = split_qkv_heads(k, num_local_heads=n_kv_local, head_dim=config.head_dim)
    v = split_qkv_heads(v, num_local_heads=n_kv_local, head_dim=config.head_dim)

    # --- RoPE on Q and K only. V is never rotated. ---
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    q = apply_rope(
        q, rope_mats, transformation_mats, False, kv_actual_global=rope_kv_actual, cluster_axis=rope_cluster_axis
    )
    k = apply_rope(
        k, rope_mats, transformation_mats, False, kv_actual_global=rope_kv_actual, cluster_axis=rope_cluster_axis
    )

    # --- attention ---
    ring_prog = program_config.get_ring_sdpa_config()
    ring_compute = program_config.get_ring_compute_kernel_config()

    # Which ring path: cache-read, or ring over this chunk's own K/V.
    #
    # `ring_joint` only takes `kv_actual_isl` (the KV-pad-aware rotation the cache read needs) when
    # the input is genuinely chunked — `Q.seq < K.seq` per device. It rejects an equal-length pair
    # with "kv_actual_isl enables KV-pad-aware rotation and requires chunked-prefill input". So a
    # chunk that has no prefix behind it AND fills the whole cache has nothing to read back and must
    # take the no-cache path, while still WRITING its K/V for whatever comes next.
    #
    # In production this only fires for a one-shot prefill whose sequence exactly fills the cache.
    # It usually cannot, because capacity is `max_seq_len` rounded UP to a whole chunk (131072 ->
    # 133120), but a chunk-aligned max_seq_len would land exactly on it — so the condition is on the
    # shapes rather than on an assumption about them.
    has_prefix_to_read = kv_cache is not None and (cached_len > 0 or tokens_global < kv_cache.capacity)

    if kv_cache is not None and write_chunk:
        # Write BEFORE choosing the path: both paths leave this chunk's K/V in the cache, which is
        # what the next chunk (and the golden-trace KV check) reads.
        write_kv_chunk(
            kv_cache,
            k,
            v,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            kv_actual=cached_len,
            sp_axis=mesh_config.sp_axis,
        )

    if not has_prefix_to_read:
        # Ring over this chunk's own SP-sharded K/V — no cache read.
        attn = dense_sp_attention_nocache(
            q,
            k,
            v,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            logical_n=logical_n,
            n_kv=config.num_kv_heads,
            head_dim=config.head_dim,
            scale=config.scaling,
            program_config=ring_prog,
            compute_kernel_config=ring_compute,
        )
    else:
        # Ring over the prefix accumulated in the cache (already written above).
        attn = dense_sp_attention(
            q,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            kv_actual=cached_len,
            logical_n=logical_n,
            n_kv=config.num_kv_heads,
            cache_global=kv_cache.capacity,
            head_dim=config.head_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            program_config=ring_prog,
            compute_kernel_config=ring_compute,
            scale=config.scaling,
            cluster_axis=mesh_config.sp_axis,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=kv_cache.num_layers,
            write_chunk=False,  # already written above, through the seam
            cache_dtype=kv_cache.k.dtype,
        )

    q.deallocate(True)
    k.deallocate(True)
    v.deallocate(True)

    # --- merge heads back and project out ---
    # [1, n_q_local, tokens, hd] -> [1, 1, tokens, n_q_local*hd]
    merged = ttnn.transpose(attn, 1, 2)
    attn.deallocate(True)
    merged = ttnn.reshape(merged, (1, 1, merged.shape[-3], n_q_local * config.head_dim))

    out = ttnn.linear(merged, weights.o_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_cfg)
    merged.deallocate(True)

    # o_proj is row-parallel: each TP chip holds a partial sum, so the collective is mandatory.
    return apply_reduce_scatter(out, mesh_config, ccl_manager, config.hidden_size)
