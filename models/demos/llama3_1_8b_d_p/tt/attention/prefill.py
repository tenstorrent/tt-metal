# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B chunked-prefill attention forward.

Copied from ``gpt_oss_d_p/tt/attention/prefill.py``. Pipeline:
QKV proj -> GQA head split -> full RoPE on Q,K -> KV-cache write -> SDPA -> concat heads ->
row-parallel o_proj -> TP all-reduce.

Every layer is full-causal, unbiased, sink-free GQA: no per-layer branch, no sliding window, no
QK-norm, no MSA/sparse path, no partial rotary.

Three SDPA paths, chosen by (SP, cached_len):

  * **SP + cache-backed ring** — chunked prefill. RingJointSDPA reads the block-cyclic SP KV cache.
    Unlike gpt-oss this is not forced by a sliding operator; Llama uses it whenever the cache has
    capacity beyond this chunk, which is the chunked-prefill case P2 needs.
  * **SP + one-shot all-gather** — a request whose whole sequence is this one chunk (Q and K/V slabs
    equal). Gather Q/K/V over SP, run plain causal SDPA, reduce-scatter the output.
  * **single-device** — the unit-test path (SP == 1), plain causal SDPA.
"""

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention
from .kv_cache import LlamaKVCache, write_kv_chunk
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


def _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, seq_len):
    """Plain causal GQA SDPA (no sinks, no sliding window)."""
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        # Pass the configured scale explicitly so it cannot drift from config.scaling; it equals the
        # kernel default (1/sqrt(head_dim)).
        scale=config.scaling,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=program_config.get_compute_kernel_config(),
    )


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
):
    """Prefill attention forward (seq_len > 1).

    Args:
        hidden_states: ``[1, 1, B*S, hidden]``
        rope_mats: ``(cos, sin)`` with llama3 scaling baked in, full head_dim wide. When
            ``indexed_rope`` is set these are the WHOLE-cache block-cyclic SP-sharded cos/sin built
            once by ``tt/rope.build_indexed_rope`` — not per-chunk.
        kv_cache: ``LlamaKVCache`` (packed K/V). Required for sequence-parallel prefill; may be None
            on the single-device unit-test path.
        user_id: cache slot index for the per-user cache write
        batch_size: number of users packed on the sequence dim
        layer_idx: this layer's index, for the per-layer cache write
        cached_len: valid prefix length already in the cache BEFORE this chunk (0 = first chunk)
        indexed_rope: use the on-device indexed RoPE

    Returns:
        ``[1, 1, B*S, hidden]``
    """
    activation_dtype = ttnn.bfloat16
    total_seq_len = hidden_states.shape[-2]
    hidden_size = config.hidden_size
    seq_len = total_seq_len // batch_size  # per-user sequence length
    if seq_len > 32 * 1024:
        activation_dtype = ttnn.bfloat8_b

    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    xqkv_fused = apply_qkv_projection(hidden_states, weights)
    hidden_states.deallocate(True)

    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # GQA: local Q heads / local KV heads per TP shard (8 and 2 at TP=4).
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    assert num_local_kv_heads >= 1, (
        f"TP={mesh_config.tp} exceeds num_kv_heads={config.num_kv_heads}; KV-head replication is not "
        f"implemented (spec known_risks: TP must divide n_kv_heads)"
    )

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Full RoPE on Q and K. indexed: the op derives this chunk's per-chip start from
    # kv_actual_global=cached_len + the device's SP mesh coord, on-device.
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    if batch_size > 1 and not indexed_rope:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats

    tt_q_orig, tt_k_orig = tt_q, tt_k
    tt_q = apply_rope(
        tt_q, rope_mats_sliced, transformation_mat, kv_actual_global=rope_kv_actual, cluster_axis=rope_cluster_axis
    )
    tt_k = apply_rope(
        tt_k, rope_mats_sliced, transformation_mat, kv_actual_global=rope_kv_actual, cluster_axis=rope_cluster_axis
    )
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Per-layer KV cache write: post-RoPE K + raw V at this chunk's offset. Single write point for
    # all chunks. tt_k / tt_v stay live (bf16) for the SDPA below; write_kv_chunk casts its own copy.
    if kv_cache is not None:
        assert isinstance(kv_cache, LlamaKVCache), "kv_cache must be a LlamaKVCache"
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
        sp = mesh_config.sp
        assert kv_cache is not None, "SP prefill needs a KV cache"
        # A non-first chunk is necessarily short-Q/long-K. Chunk 0 has that same cache-backed shape
        # only if the cache has capacity beyond its first chunk; a one-shot request where
        # max_seq_len == seq_len * sp has equal Q and K/V slabs, so it takes the gather bootstrap.
        use_cache_backed_ring = cached_len > 0 or kv_cache.max_seq_len > seq_len * sp
        if use_cache_backed_ring:
            grid = mesh_device.compute_with_storage_grid_size()
            sp_prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # carve the CCL column
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
            # fp32_dest_acc_en=False is required by the ring op's streaming-softmax compute.
            sp_kcfg = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            tt_sdpa_out = dense_sp_attention(
                tt_q,
                kv_cache.k,
                kv_cache.v,
                tt_k,
                tt_v,
                kv_actual=cached_len,
                logical_n=cached_len + seq_len * sp,
                n_kv=config.num_kv_heads,
                cache_global=kv_cache.max_seq_len,
                head_dim=config.head_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                program_config=sp_prog,
                compute_kernel_config=sp_kcfg,
                scale=config.scaling,
                cluster_axis=mesh_config.sp_axis,
                slot_idx=user_id,
                layer_idx=layer_idx,
                num_layers=kv_cache.num_layers,
                # The per-layer seam already wrote this chunk's K/V into the cache.
                write_chunk=False,
            )
        else:
            full_seq_len = seq_len * sp
            tt_q_full = mesh_config.allgather(tt_q, ccl_manager, axis=mesh_config.sp_axis, dim=2)
            tt_k_full = mesh_config.allgather(tt_k, ccl_manager, axis=mesh_config.sp_axis, dim=2)
            tt_v_full = mesh_config.allgather(tt_v, ccl_manager, axis=mesh_config.sp_axis, dim=2)
            tt_q.deallocate(True)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
            tt_q, tt_k, tt_v = tt_q_full, tt_k_full, tt_v_full
            tt_sdpa_out_full = _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, full_seq_len)
            tt_sdpa_out = ttnn.experimental.reduce_scatter_minimal_async(
                tt_sdpa_out_full,
                dim=2,
                multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=ccl_manager.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl_manager.topology,
                cluster_axis=mesh_config.sp_axis,
                barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            )
            tt_sdpa_out_full.deallocate(True)
            # reduce_scatter SUMS the SP replicas of an output each rank computed identically, so
            # divide by sp to recover it.
            tt_sdpa_out_scaled = ttnn.multiply(tt_sdpa_out, 1.0 / sp)
            ttnn.deallocate(tt_sdpa_out)
            tt_sdpa_out = tt_sdpa_out_scaled
    elif cached_len > 0:
        # Single-device chunked cache-read would need a chunk-position-aware SDPA: Q is the current
        # chunk at global offset cached_len while K/V span [0, cached_len+seq_len), so plain
        # is_causal SDPA (which assumes Q row 0 aligns with K row 0) is off by cached_len and
        # silently wrong. The SP path above is the supported chunked path; fail loud here.
        raise NotImplementedError(
            "single-device chunked cache-read attention (cached_len>0, SP==1) is not implemented — "
            "use the SP ring path (sequence_parallel=True) for chunked prefill. KV-cache "
            "storage/write is supported and validated on this path."
        )
    else:
        tt_sdpa_out = _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, seq_len)

    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    tt_sdpa_out_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out)
    tt_sdpa_out_pre_concat.deallocate(True)

    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
    tt_sdpa_out.deallocate(True)
    return apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
