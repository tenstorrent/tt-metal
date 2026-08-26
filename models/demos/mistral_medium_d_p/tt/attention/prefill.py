# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 chunked-prefill attention forward.

    QKV proj -> head split (GQA 96/8) -> full RoPE (YaRN) -> KV-cache write -> SDPA -> concat heads
    -> o_proj -> TP reduce-scatter

No biases, no attention sinks, no sliding window, no QK-norm, no partial rotary — dense causal GQA.

Sequence-parallel chunked prefill is cache-backed from the first chunk via
``dense_sp_attention`` (ring-joint SDPA over the block-cyclic SP KV cache). A one-shot request whose
cache is exactly the request length has equal-sized Q and K/V slabs; that keeps the exact
all-gather / SDPA / reduce-scatter bootstrap.
"""

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention
from .kv_cache import MistralKVCache, write_kv_chunk
from .operations import (
    apply_output_projection,
    apply_qkv_projection,
    apply_reduce_scatter,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


def _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, seq_len):
    """Single-mesh dense causal GQA SDPA (the non-SP / first-chunk path)."""
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        # Pass the scale explicitly rather than relying on the kernel default, so a config-level
        # change to `scaling` can never silently disagree with the kernel.
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
    position_idx,
    ccl_manager,
    user_id=0,
    batch_size=1,
    layer_idx=0,
    cached_len=0,
    indexed_rope=False,
):
    """Prefill attention forward. See the module docstring for the pipeline.

    Args:
        hidden_states: [1, 1, batch*seq_len, hidden_size]
        rope_mats: (cos, sin) with YaRN baked in, full head_dim wide. When ``indexed_rope`` is set
            these are the WHOLE-cache block-cyclic SP-sharded tables built once by
            ``tt/rope.build_indexed_rope``.
        kv_cache: Optional MistralKVCache. Required for SP prefill; None for single-device tests.
        cached_len: valid prefix already in the cache BEFORE this chunk (0 = first/only chunk).
        indexed_rope: use the on-device indexed RoPE.
    """
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size
    # Long chunks spill DRAM in bf16; drop the activation dtype past 32K as gpt_oss/M3 do.
    activation_dtype = ttnn.bfloat8_b if seq_len > 32 * 1024 else ttnn.bfloat16

    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}.")

    xqkv_fused = apply_qkv_projection(hidden_states, weights)
    hidden_states.deallocate(True)

    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    if batch_size > 1 and not indexed_rope:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats

    tt_q_pre, tt_k_pre = tt_q, tt_k
    tt_q = apply_rope(
        tt_q,
        rope_mats_sliced,
        transformation_mat,
        False,
        kv_actual_global=rope_kv_actual,
        cluster_axis=rope_cluster_axis,
    )
    tt_k = apply_rope(
        tt_k,
        rope_mats_sliced,
        transformation_mat,
        False,
        kv_actual_global=rope_kv_actual,
        cluster_axis=rope_cluster_axis,
    )
    tt_q_pre.deallocate(True)
    tt_k_pre.deallocate(True)

    # Per-layer KV write: post-RoPE K and raw V at this chunk's offset. Single write point for all
    # chunks; tt_k / tt_v stay live (bf16) for the SDPA below.
    if kv_cache is not None:
        assert isinstance(kv_cache, MistralKVCache), "kv_cache must be a MistralKVCache"
        write_kv_chunk(
            kv_cache,
            tt_k,
            tt_v,
            slot_idx=user_id,
            layer_idx=layer_idx,
            kv_actual=cached_len,
            sp_axis=mesh_config.sp_axis,
        )

    if config.sequence_parallel and mesh_config.sp > 1:
        sp = mesh_config.sp
        assert kv_cache is not None, "SP prefill needs a KV cache"
        # Any non-first chunk is short-Q / long-K. Chunk 0 only has that shape if the cache has
        # capacity beyond this chunk; a one-shot request where max_seq_len == seq_len*sp gives equal
        # Q and K/V slabs, which the ring reader rejects, so bootstrap it exactly instead.
        use_cache_backed_ring = cached_len > 0 or kv_cache.max_seq_len > seq_len * sp
        if use_cache_backed_ring:
            grid = mesh_device.compute_with_storage_grid_size()
            sp_prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # carve the CCL column
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
            # fp32_dest_acc_en=False is required by the ring op's streaming compute.
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
                write_chunk=False,  # the per-layer seam above already wrote it
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
            # Every SP rank computed the same full-sequence output, so the reduce-scatter summed it
            # sp times; divide it back out.
            tt_sdpa_out_scaled = ttnn.multiply(tt_sdpa_out, 1.0 / sp)
            ttnn.deallocate(tt_sdpa_out)
            tt_sdpa_out = tt_sdpa_out_scaled
    elif cached_len > 0:
        # Non-SP chunked cache-read would need a chunk-position-aware SDPA: Q sits at global offset
        # cached_len while K/V span [0, cached_len+seq_len), so plain is_causal SDPA (which assumes Q
        # row 0 aligns with K row 0) is off by cached_len and silently wrong. Fail loud instead.
        raise NotImplementedError(
            "mistral_medium_d_p: non-SP chunked cache-read (cached_len>0 with sp==1) is not implemented. "
            "KV-cache storage/write is supported; use the SP ring path (sequence_parallel=True, sp>1) "
            "for chunked prefill."
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
    # Sharded-residual contract: close with reduce-scatter, returning [1, 1, s_local, hidden/tp].
    return apply_reduce_scatter(tt_out, mesh_config, ccl_manager, hidden_size)
