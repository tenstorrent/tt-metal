# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention, dense_sp_attention_nocache
from .kv_cache import write_index_k_chunk, write_kv_chunk
from .msa import index_branch_forward, msa_sp_attention, msa_sp_attention_nocache
from .operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection,
    apply_output_projection_fused_rs,
    apply_qk_norm_per_head,
    apply_qkv_projection,
    apply_rope,
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
    user_id=0,
    batch_size=1,
    layer_idx=0,
    cached_len=0,
    indexed_rope=False,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
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
        user_id: cache slot index for the per-layer cache write
        layer_idx: this layer's index, for the per-layer cache write
        cached_len: valid prefix length already in the cache BEFORE this chunk (0 = first/only chunk).
            >0 selects the cache-read attention paths (current chunk attends the accumulated prefix).

    Returns:
        Attention output [batch, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat16
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size  # Per-user sequence length
    if seq_len > 32 * 1024:
        activation_dtype = ttnn.bfloat8_b
    else:
        activation_dtype = ttnn.bfloat16

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)
    # MSA layers need hidden_states again for the index branch (index_q/k proj); free it only for dense.
    if not config.is_sparse:
        hidden_states.deallocate(True)  # Free input activations after projection

    # NOTE (M3): QK-norm runs AFTER the head split — M3 uses per-head RMSNorm over
    # head_dim (qk_norm_type="per_head"). See the post-split block below (matches
    # transformers minimax_m3_vl: view-to-heads → q_norm/k_norm → RoPE).

    # Reshape for batch: [1, 1, B*S, QKV] -> [B, 1, S, QKV]
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # QK-norm (MiniMax-M3): per-head RMSNorm over head_dim on the split Q/K
    # ([1, n_heads, S, head_dim]), applied BEFORE RoPE, on Q and K only. The gemma (1+w)
    # fold is baked into the gain at load (weights.py); local per head (no TP reduction).
    if config.use_qk_norm and weights.q_norm is not None:
        tt_q_pre, tt_k_pre = tt_q, tt_k
        tt_q = apply_qk_norm_per_head(tt_q, weights.q_norm, config.rms_norm_eps)
        tt_k = apply_qk_norm_per_head(tt_k, weights.k_norm, config.rms_norm_eps)
        tt_q_pre.deallocate(True)
        tt_k_pre.deallocate(True)

    # Apply RoPE. indexed_rope: rope_mats are the WHOLE-cache block-cyclic SP-sharded cos/sin (built once
    # by the runtime); the indexed op derives this chunk's per-chip start from kv_actual_global=cached_len +
    # the device's cluster_axis coord on-device (no per-chunk host reshard). The per-user seq_len slice only
    # applies to the non-indexed multi-user path (indexed rope carries the whole cache, never sliced here).
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    if batch_size > 1 and not indexed_rope:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig = tt_q
    tt_k_orig = tt_k
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
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Per-layer KV cache write: post-RoPE K + raw V into the packed SP cache, at this chunk's offset
    # (cached_len). Single write point for ALL layer types and ALL chunks (the MSA index_k write is in
    # the is_sparse branch below); the cache-read attention paths below then read the accumulated prefix.
    if kv_cache is not None:
        write_kv_chunk(
            kv_cache,
            tt_k,
            tt_v,
            slot_idx=user_id,
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
        tt_iq, tt_ik = index_branch_forward(
            hidden_states,
            weights,
            rope_mats_sliced,
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
            write_index_k_chunk(
                kv_cache,
                tt_ik,
                slot_idx=user_id,
                layer_idx=layer_idx,
                kv_actual=cached_len,
                sp_axis=mesh_config.sp_axis,
            )
        if cached_len > 0:
            # Cache-read: current chunk attends the ACCUMULATED prefix. Slice this (user, layer) slot's
            # block-cyclic accumulated K/V/index_k out of the packed cache, then gather+reorder+sparse.
            sp = mesh_device.shape[mesh_config.sp_axis]
            chunk_local = seq_len  # current chunk per-chip rows
            n_chunks = cached_len // (seq_len * sp) + 1  # chunks now in the cache (incl. current)
            n_rows = n_chunks * chunk_local  # accumulated per-chip cache rows
            slot = user_id * kv_cache.num_layers + layer_idx
            # ttnn.slice on an NdShard(ROUND_ROBIN_1D) tensor corrupts the round-robin bank mapping (a
            # subsequent read then pulls the wrong banks -> the accumulated context is scrambled, chunked
            # KV-PCC craters). Convert the packed cache to plain DRAM-interleaved FIRST (the round-robin is
            # intact for the full tensor), THEN slice the slot on the interleaved result. Verified on-device
            # by test_ndshard_reorder_probe / test_msa_sp_cache_read_ndshard_pcc. Fully on-device (no host
            # round-trip); the eventual slab-aware in-kernel cache read (ring_joint-style) supersedes it.
            k_int = ttnn.to_memory_config(kv_cache.k, ttnn.DRAM_MEMORY_CONFIG)
            v_int = ttnn.to_memory_config(kv_cache.v, ttnn.DRAM_MEMORY_CONFIG)
            ik_int = ttnn.to_memory_config(kv_cache.index_k, ttnn.DRAM_MEMORY_CONFIG)
            k_acc = ttnn.slice(k_int, (slot, 0, 0, 0), (slot + 1, 1, n_rows, config.head_dim))
            v_acc = ttnn.slice(v_int, (slot, 0, 0, 0), (slot + 1, 1, n_rows, config.head_dim))
            ik_acc = ttnn.slice(ik_int, (slot, 0, 0, 0), (slot + 1, 1, n_rows, config.head_dim))
            k_int.deallocate(True)
            v_int.deallocate(True)
            ik_int.deallocate(True)
            tt_sdpa_out = msa_sp_attention(
                tt_q,
                k_acc,
                v_acc,
                tt_iq,
                ik_acc,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                cached_len=cached_len,
                s_local=seq_len,
                n_chunks=n_chunks,
                chunk_local=chunk_local,
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
            tt_sdpa_out = dense_sp_attention(
                tt_q,
                kv_cache.k,
                kv_cache.v,
                tt_k,
                tt_v,
                kv_actual=cached_len,
                logical_n=logical_n,
                n_kv=config.num_kv_heads,
                # Ring-gather output buffer must span the FULL cache capacity: ring_joint gathers the
                # entire per-device cache shard (seq_local = max_seq_len/sp rows -> x sp across the ring =
                # max_seq_len), independent of the valid prefix (logical_n/kv_actual_isl drive causal
                # masking of the not-yet-written tail). Sizing it to logical_n only worked for the 2-chunk
                # case where the last chunk's logical_n == max_seq_len; any run with >2 chunks (e.g. 50k /
                # 11 chunks) fails "gather dim 2 too small: got <logical_n>, expected >= max_seq_len".
                cache_global=kv_cache.max_seq_len,
                head_dim=config.head_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                program_config=sp_prog,
                compute_kernel_config=sp_kcfg,
                scale=config.head_dim**-0.5,
                cluster_axis=mesh_config.sp_axis,
                slot_idx=user_id,
                layer_idx=layer_idx,
                num_layers=kv_cache.num_layers,
                write_chunk=False,
            )
        else:
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
    tt_sdpa_out = concat_heads(tt_sdpa_out)
    tt_sdpa_out_pre_concat.deallocate(True)

    # Flatten back for output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection + tensor-parallel allreduce.
    # When TP > 1 we use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay as separate ops. See
    # apply_output_projection_fused_rs for the per-shape tuned configs.
    # The fused MM+RS op (minimal_matmul_strided_reduce_scatter_async) ONLY supports Ring topology, so
    # fall back to the plain o_proj + all-reduce under Linear (e.g. the single-galaxy FABRIC_1D mesh).
    use_fused_rs = (
        mesh_config.tp > 1
        and is_shape_fused_mm_rs_supported(tt_sdpa_out)
        and ccl_manager.topology == ttnn.Topology.Ring
    )
    if use_fused_rs:
        rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    else:
        tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
    return tt_out_result
