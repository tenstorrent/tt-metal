# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.common.prefill.prefill_chunk import PrefillChunk
from models.demos.minimax_m3.utils.profiler_utils import FINE, zone

from ..residual import use_sharded_residual
from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention, dense_sp_attention_nocache
from .kv_cache import write_index_k_chunk, write_kv_chunk
from .msa import index_branch_forward, msa_sp_attention_cache_read, msa_sp_attention_nocache
from .operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection,
    apply_output_projection_fused_rs,
    apply_qk_norm_per_head,
    apply_qkv_projection,
    apply_reduce_scatter,
    apply_rope,
    assert_sharded_residual_unpadded,
    concat_heads,
    is_shape_fused_mm_rs_supported,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


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
    position_idx,
    ccl_manager,
    layer_idx=0,
    indexed_rope=False,
    chunks=None,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    Args:
        hidden_states: Input tensor [1, 1, seq_len, hidden_size]
        rope_mats: Tuple of (cos, sin) matrices for RoPE
        weights: Attention weights
        kv_cache: Externally-owned MiniMaxKVCache (packed K/V/index_k)
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        position_idx: Position indices (unused in prefill)
        ccl_manager: Communication manager
        layer_idx: this layer's index, for the per-layer cache write
        chunks: the forward's PrefillChunks (common/prefill/prefill_chunk.py), one equal block of rows each; None = one
            chunk in slot 0 at position 0. Projections run once on all rows; the attention core runs per non-empty
            chunk against its slot_id at its actual_start (>0 selects the cache-read paths). Empty chunks (the
            tail) get zero rows. A partial chunk's pad rows follow its real rows, so causality keeps them out of
            the real rows' attention; their K/V land past the slot's valid prefix, which later chunks overwrite.

    Returns:
        Attention output [1, 1, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat16
    seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    if seq_len > 32 * 1024:
        activation_dtype = ttnn.bfloat8_b
    else:
        activation_dtype = ttnn.bfloat16

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # QKV projection
    with zone("qkv_proj"):
        xqkv_fused = apply_qkv_projection(hidden_states, weights)
    # MSA layers need hidden_states again for the index branch (index_q/k proj); free it only for dense.
    if not config.is_sparse:
        hidden_states.deallocate(True)  # Free input activations after projection

    # NOTE (M3): QK-norm runs AFTER the head split — M3 uses per-head RMSNorm over
    # head_dim (qk_norm_type="per_head"). See the post-split block below (matches
    # transformers minimax_m3_vl: view-to-heads → q_norm/k_norm → RoPE).

    def core(xqkv, hidden, chunk, rows):
        return _attention_core(
            xqkv,
            hidden,
            rope_mats,
            weights,
            kv_cache,
            config,
            mesh_config,
            mesh_device,
            program_config,
            transformation_mat,
            ccl_manager,
            slot_id=chunk.slot_id,
            layer_idx=layer_idx,
            cached_len=chunk.actual_start,
            indexed_rope=indexed_rope,
            seq_len=rows,
        )

    chunks = chunks or [PrefillChunk(slot_id=0, actual_start=0, actual_end=seq_len)]
    if len(chunks) == 1:
        tt_sdpa_out = core(xqkv_fused, hidden_states, chunks[0], seq_len)
    else:
        assert indexed_rope, "a packed forward needs the whole-cache indexed RoPE"
        rows = seq_len // len(chunks)
        real = [c for c in chunks if not c.is_empty]

        def chunk_rows(t, k):
            return ttnn.slice(t, (0, 0, k * rows, 0), (1, 1, (k + 1) * rows, t.shape[-1]))

        outs = [
            core(chunk_rows(xqkv_fused, k), chunk_rows(hidden_states, k) if config.is_sparse else None, c, rows)
            for k, c in enumerate(real)
        ]
        xqkv_fused.deallocate(True)
        if config.is_sparse:
            hidden_states.deallocate(True)
        # Empty chunks (the tail) get zero rows; concatenating all chunks keeps one concat program per forward width.
        empties = len(chunks) - len(real)
        zeros = ttnn.zeros_like(outs[0]) if empties else None
        tt_sdpa_out = ttnn.concat(outs + [zeros] * empties, dim=2)
        for o in outs + ([zeros] if empties else []):
            o.deallocate(True)

    return _output_projection(tt_sdpa_out, weights, mesh_config, ccl_manager, hidden_size, activation_dtype)


def _attention_core(
    xqkv_fused,
    hidden_states,
    rope_mats,
    weights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    program_config,
    transformation_mat,
    ccl_manager,
    slot_id,
    layer_idx,
    cached_len,
    indexed_rope,
    seq_len,
):
    """One chunk's rows, slot ``slot_id`` at position ``cached_len`` (the chunk's actual_start: the prefix already
    cached): split heads -> QK-norm -> RoPE -> KV write -> attention -> concat heads. Consumes xqkv_fused (and
    hidden_states on MSA layers)."""
    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    with zone("split_heads", FINE):
        tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # QK-norm (MiniMax-M3): per-head RMSNorm over head_dim on the split Q/K
    # ([1, n_heads, S, head_dim]), applied BEFORE RoPE, on Q and K only. The gemma (1+w)
    # fold is baked into the gain at load (weights.py); local per head (no TP reduction).
    if config.use_qk_norm and weights.q_norm is not None:
        with zone("qk_norm", FINE):
            tt_q_pre, tt_k_pre = tt_q, tt_k
            tt_q = apply_qk_norm_per_head(tt_q, weights.q_norm, config.rms_norm_eps)
            tt_k = apply_qk_norm_per_head(tt_k, weights.k_norm, config.rms_norm_eps)
            tt_q_pre.deallocate(True)
            tt_k_pre.deallocate(True)

    # Apply RoPE. indexed_rope: rope_mats are the WHOLE-cache block-cyclic SP-sharded cos/sin (built once
    # by the runtime); the indexed op derives this chunk's per-chip start from kv_actual_global=cached_len +
    # the device's cluster_axis coord on-device (no per-chunk host reshard).
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    with zone("rope", FINE):
        tt_q = apply_rope(
            tt_q,
            rope_mats,
            transformation_mat,
            is_decode_mode=False,
            kv_actual_global=rope_kv_actual,
            cluster_axis=rope_cluster_axis,
        )
        tt_k = apply_rope(
            tt_k,
            rope_mats,
            transformation_mat,
            is_decode_mode=False,
            kv_actual_global=rope_kv_actual,
            cluster_axis=rope_cluster_axis,
        )
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Per-layer KV cache write: post-RoPE K + raw V into the packed SP cache, at this chunk's offset
    # (cached_len). Single write point for ALL layer types and ALL chunks (the MSA index_k write is in
    # the is_sparse branch below); the cache-read attention paths below then read the accumulated prefix.
    if kv_cache is not None:
        with zone("kv_write", FINE):
            write_kv_chunk(
                kv_cache,
                tt_k,
                tt_v,
                slot_id=slot_id,
                layer_idx=layer_idx,
                kv_actual=cached_len,
                sp_axis=mesh_config.sp_axis,
            )

    # Attention core — per-layer gate (config.is_sparse from M3 sparse_attention_freq):
    #   MSA layers (3-59): index branch (index_q/k proj -> norm -> RoPE) + block-sparse SP attention
    #     (msa_sp_attention_nocache): AllGather K/V/index_k across SP, keep q/index_q SP-sharded
    #     (S/sp rows/device), per-device causality via mesh-coord cluster_axis (cached_len + rank*S_local) -> SP-sharded
    #     output. num_groups = local KV heads (1 GQA group/KV head; 1 at TP=4). Degenerates to the
    #     full-context path at sp=1. cached_len is 0 for the first chunk (multi-chunk cache: TODO).
    #   Dense layers (0-2): plain causal GQA SDPA (exact at sp=1). SP dense via dense_sp_attention
    #     (ring_joint, dense_sp.py) + the per-layer KV cache lifecycle is the remaining model-level wiring.
    if config.is_sparse:
        with zone("index_branch"):
            tt_iq, tt_ik = index_branch_forward(
                hidden_states,
                weights,
                rope_mats,
                transformation_mat,
                index_dim=config.msa_index_dim,
                rms_norm_eps=config.rms_norm_eps,
                kv_actual_global=rope_kv_actual,
                cluster_axis=rope_cluster_axis,
            )
        hidden_states.deallocate(True)
        # MSA-only: cache the post-norm/post-RoPE index_k (single shared head, TP-replicated) at this
        # chunk's offset, so a later chunk's cache-read can score against the accumulated context.
        if kv_cache is not None:
            with zone("index_k_write", FINE):
                write_index_k_chunk(
                    kv_cache,
                    tt_ik,
                    slot_id=slot_id,
                    layer_idx=layer_idx,
                    kv_actual=cached_len,
                    sp_axis=mesh_config.sp_axis,
                )
        # The exact-size path's top-k needs at least topk_blocks block columns, so a cold chunk shorter than the
        # top-k window also takes the cache-read path, whose top-k is bounded by kv_len.
        chunk_global = seq_len * mesh_device.shape[mesh_config.sp_axis]
        topk_window = config.msa_topk_blocks * config.msa_block_size
        if cached_len > 0 or chunk_global < topk_window:
            assert kv_cache is not None, (
                f"MSA chunk of {chunk_global} tokens at cached_len={cached_len} needs a KV cache "
                f"(a cold chunk reads it when shorter than the {topk_window}-token top-k window)"
            )
            # Cache-read: the current chunk attends the accumulated prefix, gathered across SP straight
            # from this (slot_id, layer) row of the packed cache (msa_sp_attention_cache_read). cached_len may be
            # mid-slab (a 32-aligned multi-turn resume); the read sizes the gather for the fullest rank.
            cache_row = slot_id * kv_cache.num_layers + layer_idx
            tt_sdpa_out = msa_sp_attention_cache_read(
                tt_q,
                tt_iq,
                kv_cache,
                cache_row=cache_row,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                cached_len=cached_len,
                chunk_local=seq_len,  # current chunk per-chip rows
                scale=config.head_dim**-0.5,
                block_size=config.msa_block_size,
                topk_blocks=config.msa_topk_blocks,
                num_groups=num_local_kv_heads,
            )
        else:
            tt_sdpa_out = msa_sp_attention_nocache(
                tt_q,
                tt_k,
                tt_v,
                tt_iq,
                tt_ik,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                cached_len=0,
                s_local=seq_len,
                scale=config.head_dim**-0.5,
                block_size=config.msa_block_size,
                topk_blocks=config.msa_topk_blocks,
                num_groups=num_local_kv_heads,
            )
    elif config.sequence_parallel:
        # SP dense (first chunk, no prior cache): ring_joint over the chunk's own SP-sharded K/V, each
        # device's query shard attending the full sequence reconstructed across the SP ring. q/k/v are
        # the per-device shards (seq_len = S/sp rows). logical_n = full sequence = seq_len * sp.
        sp = mesh_device.shape[mesh_config.sp_axis]
        grid = mesh_device.compute_with_storage_grid_size()
        sp_prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # carve the CCL column
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,  # Pavle's minimax3_gqa_causal_perf
        )
        sp_kcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        if cached_len > 0:
            # Cache-read: ring_joint over the accumulated prefix in the cache (the seam already wrote this
            # chunk -> write_chunk=False). logical_n = full valid prefix = cached_len + this chunk.
            logical_n = cached_len + seq_len * sp
            with zone("ring_joint_sdpa"):
                tt_sdpa_out = dense_sp_attention(
                    tt_q,
                    kv_cache.k,
                    kv_cache.v,
                    tt_k,
                    tt_v,
                    kv_actual=cached_len,
                    logical_n=logical_n,
                    n_kv=config.num_kv_heads,
                    # The ring-gather output buffer is sized for the full cache capacity (max_seq_len): the op
                    # checks it against the cache shard ("gather dim 2 too small"), although it only moves the
                    # valid prefix (logical_n / kv_actual drive that and the causal mask).
                    cache_global=kv_cache.max_seq_len,
                    head_dim=config.head_dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    program_config=sp_prog,
                    compute_kernel_config=sp_kcfg,
                    scale=config.head_dim**-0.5,
                    cluster_axis=mesh_config.sp_axis,
                    slot_idx=slot_id,
                    layer_idx=layer_idx,
                    num_layers=kv_cache.num_layers,
                    write_chunk=False,
                )
        else:
            with zone("ring_joint_sdpa"):
                tt_sdpa_out = dense_sp_attention_nocache(
                    tt_q,
                    tt_k,
                    tt_v,
                    mesh_config=mesh_config,
                    ccl_manager=ccl_manager,
                    logical_n=seq_len * sp,
                    n_kv=config.num_kv_heads,
                    head_dim=config.head_dim,
                    scale=config.head_dim**-0.5,
                    program_config=sp_prog,
                    compute_kernel_config=sp_kcfg,
                )
    else:
        with zone("sdpa"):
            tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
                compute_kernel_config=program_config.get_compute_kernel_config(),
            )
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out_pre_concat = tt_sdpa_out
    with zone("concat_heads", FINE):
        tt_sdpa_out = concat_heads(tt_sdpa_out)
    tt_sdpa_out_pre_concat.deallocate(True)
    return tt_sdpa_out


def _output_projection(tt_sdpa_out, weights, mesh_config, ccl_manager, hidden_size, activation_dtype):
    # Output projection + the closing TP collective. Under a SHARDED residual the collective is a
    # reduce-scatter only (o_proj is row-parallel, so the RS both completes the partial sums and lands
    # the result already in the residual's emb/tp layout); under the replicated residual it must be a
    # full all-reduce.
    # When TP > 1 we use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay as separate ops. See
    # apply_output_projection_fused_rs for the per-shape tuned configs.
    # The fused MM+RS op (minimal_matmul_strided_reduce_scatter_async) ONLY supports Ring topology, so
    # fall back to the plain o_proj + all-reduce under Linear (e.g. the single-galaxy FABRIC_1D mesh).
    sharded_residual = use_sharded_residual() and mesh_config.tp > 1
    use_fused_rs = (
        mesh_config.tp > 1
        and is_shape_fused_mm_rs_supported(tt_sdpa_out)
        and ccl_manager.topology == ttnn.Topology.Ring
    )
    if use_fused_rs:
        with zone("o_proj_fused_rs"):
            rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        if sharded_residual:
            # The fused op already reduce-scattered: that IS the sharded-residual output. Only the
            # padding trim would remain, and a sharded residual admits no padding, so nothing is left.
            assert_sharded_residual_unpadded(mesh_config, hidden_size)
            return rs_out
        with zone("ccl_out_allgather"):
            tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    else:
        with zone("o_proj"):
            tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
        tt_sdpa_out.deallocate(True)
        if sharded_residual:
            with zone("ccl_out_reduce_scatter"):
                return apply_reduce_scatter(tt_out, mesh_config, ccl_manager, hidden_size)
        with zone("ccl_out_allreduce"):
            tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
    return tt_out_result
