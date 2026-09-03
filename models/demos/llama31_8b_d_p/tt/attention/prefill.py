# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B prefill attention forward: GQA + full RoPE + causal SDPA + o_proj + TP collective.

Template: ``models/demos/gpt_oss_d_p/tt/attention/prefill.py:51`` (``attention_forward``), ``:34``
(``_run_sdpa``), ``:40`` (``is_causal=True``), ``:43`` (``scale=config.scaling``), ``:116`` (the qkv
projection), ``:127`` (the head split), ``:143`` / ``:151`` (RoPE on Q / K), ``:168``
(``write_kv_chunk``), ``:184`` (the SP branch), ``:272`` (the single-card SDPA), ``:280``
(``concat_heads``), ``:302`` (``o_proj``), ``:304`` (``apply_allreduce``).

Pipeline, per chip (``S_loc = S/sp``; the numbers are TP=8 / SP=4, scheme A)::

    hidden_states [1,1,S_loc,4096] bf16 TILE
      -> q [1,1,S_loc,512] , kv [1,1,S_loc,256]                 (3 linears + 1 concat, DEC-011)
      -> Q [1,4,S_loc,128] , K [1,1,S_loc,128] , V [1,1,S_loc,128]
      -> full RoPE on Q and K (shape-preserving)
      -> write_kv_chunk(K_post_rope, V)                          (when kv_cache is not None)
      -> SDPA (is_causal=True, scale=config.scaling)  -> [1,4,S_loc,128]
         or dense_sp_attention (SP>1, cache-backed)   -> [1,4,S_loc,128]      (P8)
      -> concat_heads                                 -> [1,1,S_loc,512]
      -> o_proj                                       -> [1,1,S_loc,4096]  (partial sum)
      -> TP all-reduce                                -> [1,1,S_loc,4096]

**GQA needs no on-chip KV repeat.** ``ttnn.transformer.scaled_dot_product_attention`` handles the
group itself; the only head constraint is
``ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98``
``TT_FATAL(nqh >= nkv && nqh % nkv == 0, ...)``. At TP=1 (P5): ``32 >= 8 && 32 % 8 == 0``; at TP=8:
``4 >= 1 && 4 % 1 == 0``. Appendix D is explicit about this and about the two SDPA arguments Llama
must **drop** vs the template: ``sliding_window_size=`` (``:44``) and ``attention_sink=`` (``:45``).

**Deletions vs the template** (``03_OUTLINE.md`` §3.10): the two SDPA arguments above, and the whole
fused matmul+reduce-scatter branch (``:292-300``) — it is gated off on Blackhole anyway because the
op races there. The ``batch_size > 1`` reshapes (``:120-121``, ``:284-285``) **stay**: the runtime's
multi-user path needs them.

**``cached_len > 0`` on a single device stays a loud ``NotImplementedError``** (verbatim in spirit
from ``:266-270``). A plain ``is_causal`` SDPA assumes Q row 0 aligns with K row 0, so with a
non-empty cache it is off by ``cached_len`` and **silently wrong**. Reading the cache back for
attention is the SP ring path (``dense_sp.py``, P8) or a paged chunked SDPA — not a single-card path.
"""

from __future__ import annotations

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention
from .kv_cache import LlamaKVCache, write_kv_chunk
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_reduce_scatter,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# 03_OUTLINE.md §1 convention 11: bf16 activations, bf8_b only above this sequence length.
_BF8_ACTIVATION_SEQ_THRESHOLD = 32 * 1024


def _run_sdpa(tt_q, tt_k, tt_v, config, program_config, compute_kernel_config, mesh_device, seq_len):
    """Single-chip GQA causal SDPA. No sliding window, no attention sink (Llama has neither)."""
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        # Passed explicitly so the QK scale is visible at the call site; it equals the kernel
        # default 1/sqrt(head_dim), and config.scaling is the single source of truth for it.
        scale=config.scaling,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=compute_kernel_config,
    )


def _run_sp_bootstrap_sdpa(
    tt_q, tt_k, tt_v, config, program_config, compute_kernel_config, mesh_device, mesh_config, ccl_manager, seq_len
):
    """The SP **one-shot** attention core: all-gather Q/K/V on the SP axis -> plain causal SDPA ->
    reduce-scatter -> ``x 1/sp`` (``DEC-021``; template
    ``models/demos/gpt_oss_d_p/tt/attention/prefill.py:233``).

    Why it exists at all: the ring path needs Q strictly shorter than the per-chip cache shard
    (``ring_joint_sdpa_device_operation.cpp:580``), so a request whose cache is exactly one chunk
    long — ``max_seq_len == seq_len * sp``, which is what a one-shot ``G-MESH-KV`` run looks like —
    has no ring to take, chunk 0 included. It is also the only SP path that does not depend on the
    KV cache being correct, which makes it the bisection tool when a cache-backed run fails.

    Why the ``x 1/sp``: after the all-gather every SP row holds the *same* full-sequence Q/K/V and so
    computes the *same* full output. The reduce-scatter is being used as a scatter, and it sums
    ``sp`` identical copies on the way, so the rescale undoes the sum. Routed through
    ``mesh_config.reduce_scatter`` rather than a raw ``reduce_scatter_minimal_async`` — the
    "collectives only via ``MeshConfig``" convention, and the clean-up ``DEC-021`` owed to P8.
    """
    sp = mesh_config.sp
    sp_axis = mesh_config.sp_axis
    full_seq_len = seq_len * sp
    gathered = [mesh_config.allgather(t, ccl_manager, axis=sp_axis, dim=2) for t in (tt_q, tt_k, tt_v)]
    tt_out_full = _run_sdpa(
        gathered[0], gathered[1], gathered[2], config, program_config, compute_kernel_config, mesh_device, full_seq_len
    )
    for t in gathered:
        t.deallocate(True)
    scattered = mesh_config.reduce_scatter(tt_out_full, ccl_manager, dim=2, axis=sp_axis)
    tt_out_full.deallocate(True)
    rescaled = ttnn.multiply(scattered, 1.0 / sp)
    scattered.deallocate(True)
    return rescaled


def attention_forward(
    hidden_states,
    rope_mats,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    ccl_manager,
    user_id=0,
    batch_size=1,
    layer_idx=0,
    cached_len=0,
    indexed_rope=False,
    scatter_output=False,
    compute_kernel_config=None,
):
    """Prefill attention forward (``seq_len > 1``).

    Args:
        hidden_states: ``[1, 1, batch*S_loc, hidden]`` bf16 TILE.
        rope_mats: ``(cos, sin)``, Meta/interleaved, llama3 scaling already baked in. Per-chunk
            tables from ``tt/rope.build_prefill_rope``, or the whole-cache block-cyclic SP tables
            from ``build_indexed_rope`` when ``indexed_rope`` is set.
        weights: :class:`~.weights.AttentionWeights`.
        kv_cache: :class:`~.kv_cache.LlamaKVCache` or ``None`` (the single-device unit-test path).
        config: :class:`~.config.AttentionConfig`.
        mesh_config: ``MeshConfig``.
        mesh_device: the ttnn mesh device.
        program_config: :class:`~.config.ProgramConfig`.
        transformation_mat: the ``[1, 1, 32, 32]`` RoPE matrix.
        ccl_manager: ``CCLManager``; only touched when TP > 1 or SP > 1.
        user_id: cache slot index for the per-user cache write.
        batch_size: number of users packed on the sequence dim.
        layer_idx: this layer's index, for the per-layer cache write.
        cached_len: valid prefix already in the cache BEFORE this chunk (0 = first/only chunk).
        indexed_rope: use the on-device indexed RoPE (whole-cache block-cyclic SP cos/sin).
        scatter_output: ``False`` (scheme A) -> TP all-reduce, full hidden out. ``True`` (scheme B)
            -> reduce-scatter only, ``hidden/tp`` out (``DEC-018``).
        compute_kernel_config: for the projections and SDPA. ``None`` builds it from
            ``program_config`` (``DEC-031``).

    Returns:
        ``[1, 1, batch*S_loc, hidden]`` (scheme A) or ``[1, 1, batch*S_loc, hidden/tp]`` (scheme B).
    """
    total_seq_len = hidden_states.shape[-2]
    seq_len = total_seq_len // batch_size  # per-user sequence length
    activation_dtype = ttnn.bfloat8_b if seq_len > _BF8_ACTIVATION_SEQ_THRESHOLD else ttnn.bfloat16

    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. There is no decode path here.")

    if compute_kernel_config is None:
        compute_kernel_config = program_config.get_compute_kernel_config(mesh_device)

    # --- QKV projection: three linears + one K|V concat (DEC-011) ---
    tt_q_flat, tt_kv_flat = apply_qkv_projection(hidden_states, weights, compute_kernel_config)
    hidden_states.deallocate(True)  # free the input activations before the head split allocates

    # Reshape for batch: [1, 1, B*S, X] -> [B, 1, S, X]
    if batch_size > 1:
        tt_q_flat = ttnn.reshape(tt_q_flat, [batch_size, 1, seq_len, -1])
        tt_kv_flat = ttnn.reshape(tt_kv_flat, [batch_size, 1, seq_len, -1])

    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(tt_q_flat, tt_kv_flat, num_local_heads, num_local_kv_heads)
    tt_q_flat.deallocate(True)
    tt_kv_flat.deallocate(True)

    # --- Full RoPE on Q and K ---
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    if batch_size > 1 and not indexed_rope:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_pre_rope, tt_k_pre_rope = tt_q, tt_k
    tt_q = apply_rope(
        tt_q,
        rope_mats_sliced,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=rope_kv_actual,
        cluster_axis=rope_cluster_axis,
    )
    tt_k = apply_rope(
        tt_k,
        rope_mats_sliced,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=rope_kv_actual,
        cluster_axis=rope_cluster_axis,
    )
    tt_q_pre_rope.deallocate(True)
    tt_k_pre_rope.deallocate(True)

    # --- Per-layer KV cache write: post-RoPE K + raw V at this chunk's offset ---
    # tt_k / tt_v stay live (bf16) for the SDPA that follows; write_kv_chunk casts its own copy.
    if kv_cache is not None:
        assert isinstance(kv_cache, LlamaKVCache), f"kv_cache must be a LlamaKVCache, got {type(kv_cache).__name__}"
        write_kv_chunk(
            kv_cache,
            tt_k,
            tt_v,
            slot_idx=user_id,
            layer_idx=layer_idx,
            kv_actual=cached_len,
            sp_axis=mesh_config.sp_axis,
        )

    # --- Attention core ---
    if config.sequence_parallel and mesh_config.sp > 1:
        assert kv_cache is not None, "SP prefill needs a KV cache"
        # DEC-021's selection rule, verbatim from the template (`gpt_oss prefill.py:191`): the ring
        # path needs Q shorter than the per-chip cache shard, so a cache sized to exactly one chunk
        # takes the bootstrap even for chunk 0. Production sizes max_seq_len above one chunk and
        # therefore always takes the ring.
        if cached_len > 0 or kv_cache.max_seq_len > seq_len * mesh_config.sp:
            tt_sdpa_out = dense_sp_attention(
                tt_q,
                kv_cache.k,
                kv_cache.v,
                tt_k,
                tt_v,
                kv_actual=cached_len,
                logical_n=cached_len + seq_len * mesh_config.sp,
                n_kv=config.num_kv_heads,
                cache_global=kv_cache.max_seq_len,
                head_dim=config.head_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                program_config=program_config,
                scale=config.scaling,
                cluster_axis=mesh_config.sp_axis,
                slot_idx=user_id,
                layer_idx=layer_idx,
                num_layers=kv_cache.num_layers,
                write_chunk=False,  # the per-layer seam already wrote this chunk above
            )
        else:
            tt_sdpa_out = _run_sp_bootstrap_sdpa(
                tt_q,
                tt_k,
                tt_v,
                config,
                program_config,
                compute_kernel_config,
                mesh_device,
                mesh_config,
                ccl_manager,
                seq_len,
            )
    elif cached_len > 0:
        raise NotImplementedError(
            "llama31_8b_d_p: single-device chunked cache-read attention (cached_len>0) is not "
            "implemented. KV-cache storage and the chunk write ARE implemented and gated "
            "(G-KV / test_kv_cache_vs_ref.py), but reading the accumulated prefix back needs a "
            "chunk-position-aware SDPA: Q is the current chunk at global offset cached_len while "
            "K/V span [0, cached_len+seq_len), so a plain is_causal SDPA (which assumes Q row 0 "
            "aligns with K row 0) is off by cached_len and SILENTLY WRONG. The correct paths are "
            "the paged chunked_scaled_dot_product_attention or the SP ring-joint SDPA over the "
            "block-cyclic cache (tt/attention/dense_sp.py, P8)."
        )
    else:
        tt_sdpa_out = _run_sdpa(tt_q, tt_k, tt_v, config, program_config, compute_kernel_config, mesh_device, seq_len)

    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # --- Concat heads back to the (local) hidden dim ---
    tt_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out)
    tt_pre_concat.deallocate(True)

    # Flatten back for the output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # --- o_proj + the TP collective ---
    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype, compute_kernel_config)
    tt_sdpa_out.deallocate(True)
    if scatter_output:
        return apply_reduce_scatter(tt_out, mesh_config, ccl_manager)
    return apply_allreduce(tt_out, mesh_config, ccl_manager)
