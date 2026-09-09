# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel attention over durable ring KV caches."""

import ttnn

from .global_kv_cache import GLOBAL_HEAD_DIM, GLOBAL_ROTARY_DIM, pack_global_kv_device
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    concat_heads,
    prefill_short_lived_memcfg,
    split_qkv_heads_prefill,
)
from .ring_prefill import (
    PackedRingKVCache,
    ring_packed_prefill_attention,
    ring_prefill_attention,
    write_chunk_to_packed_ring_cache,
    write_chunk_to_ring_cache,
)
from .weights import AttentionWeights


def prefill_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    config,
    mesh_config,
    ccl_manager,
    ring_kv_cache,
    ring_max_seq_len,
    chunk_start_idx=0,
    ring_layer_idx=0,
    ring_num_layers=1,
    packed_global_rope=None,
    packed_sliding_rope=None,
):
    """Write a user's chunk and attend its cached prefix."""
    if ring_kv_cache is None:
        raise ValueError("Galaxy prefill requires a ring KV cache")
    tp = mesh_config.tp
    chunk_offset = int(chunk_start_idx)
    kv_tied = weights.is_global
    xqkv = apply_qkv_projection(hidden_states, weights, kv_tied=kv_tied)

    # Short-lived prefill activations in L1 when GEMMA4_PREFILL_L1_ACT=1 (Qwen36
    # #48861). o_proj / allreduce stay DRAM (CB clash with CCL).
    act_mc = prefill_short_lived_memcfg()
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv,
        config,
        weights.is_global,
        tp=tp,
        kv_replicated=weights.kv_replicated,
        memory_config=act_mc,
        kv_tied=kv_tied,
    )

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc)

    packed_global_ring = weights.is_global and isinstance(ring_kv_cache, PackedRingKVCache)
    packed_sliding_ring = config.is_sliding and ring_kv_cache is not None
    if weights.is_global:
        # The tied projection is one semantic KV value. Normalize it once without
        # gamma: this entire 512-wide result is V. K branches from this value;
        # packed-only serving transforms just its active rotary quarter below.
        tt_k.deallocate(True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=act_mc)
        tt_k = None
    else:
        tt_k = apply_per_head_norm(
            tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc
        )
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=act_mc)

    # Apply RoPE to Q and the rotary part of K.
    if packed_global_ring:
        if packed_global_rope is None:
            raise RuntimeError("packed global ring attention requires pre-gathered packed RoPE tensors")
        q_cos, q_sin, _, _, trans_mat = packed_global_rope
        q_full = tt_q
        q_rotary = ttnn.slice(
            q_full,
            (0, 0, 0, 0),
            tuple(q_full.shape)[:-1] + (GLOBAL_ROTARY_DIM,),
            memory_config=act_mc,
        )
        q_nonrotary = ttnn.slice(
            q_full,
            (0, 0, 0, GLOBAL_ROTARY_DIM),
            tuple(q_full.shape)[:-1] + (GLOBAL_HEAD_DIM,),
            memory_config=act_mc,
        )
        q_rotated = ttnn.experimental.rotary_embedding_llama(
            q_rotary, q_cos, q_sin, trans_mat, is_decode_mode=False, memory_config=act_mc
        )
        tt_q = ttnn.concat((q_rotated, q_nonrotary), dim=-1, memory_config=act_mc)
        for tensor in (q_full, q_rotary, q_nonrotary, q_rotated):
            tensor.deallocate(True)
    elif packed_sliding_ring:
        if packed_sliding_rope is None:
            raise RuntimeError("packed sliding ring attention requires pre-gathered adjacent RoPE tensors")
        sliding_cos, sliding_sin, trans_mat = packed_sliding_rope
        q_unrotated = tt_q
        tt_q = ttnn.experimental.rotary_embedding_llama(
            q_unrotated, sliding_cos, sliding_sin, trans_mat, is_decode_mode=False, memory_config=act_mc
        )
        q_unrotated.deallocate(True)
        k_unrotated = tt_k
        tt_k = ttnn.experimental.rotary_embedding_llama(
            k_unrotated, sliding_cos, sliding_sin, trans_mat, is_decode_mode=False, memory_config=act_mc
        )
        k_unrotated.deallocate(True)
    sliding_window = config.sliding_window
    if packed_global_ring:
        packed_q = tt_q
        packed_kv = pack_global_kv_device(
            tt_v,
            weights.k_norm_rotary_weight,
            cos_cache,
            sin_cache,
            canonical_k=tt_k,
            packed_rope_mats=packed_global_rope,
            value_is_packed=True,
            memory_config=act_mc,
        )
        write_chunk_to_packed_ring_cache(
            ring_kv_cache.kv,
            packed_kv,
            mesh_config,
            kv_actual_global=chunk_offset,
            layer_idx=ring_layer_idx,
            num_layers=ring_num_layers,
            ccl_manager=ccl_manager,
        )
    else:
        packed_q = None
        write_chunk_to_ring_cache(
            ring_kv_cache[0],
            ring_kv_cache[1],
            tt_k,
            tt_v,
            mesh_config,
            kv_actual_global=chunk_offset,
            layer_idx=ring_layer_idx,
            num_layers=ring_num_layers,
            ccl_manager=ccl_manager,
        )
    cp_ring_ckc = ttnn.init_device_compute_kernel_config(
        tt_q.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    num_local_kv_heads_ring = tt_v.shape[1]
    ring_logical_n = ring_max_seq_len
    if packed_global_ring:
        tt_sdpa = ring_packed_prefill_attention(
            packed_q,
            ring_kv_cache.kv,
            mesh_device=tt_q.device(),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            num_local_kv_heads=num_local_kv_heads_ring,
            max_seq_len=ring_max_seq_len,
            logical_n=ring_logical_n,
            kv_actual_global=chunk_offset,
            scale=1.0,
            compute_kernel_config=cp_ring_ckc,
            layer_idx=ring_layer_idx,
            num_layers=ring_num_layers,
        )
        packed_kv.deallocate(True)
    else:
        tt_sdpa = ring_prefill_attention(
            tt_q,
            ring_kv_cache[0],
            ring_kv_cache[1],
            mesh_device=tt_q.device(),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            num_local_kv_heads=num_local_kv_heads_ring,
            head_dim=config.head_dim,
            max_seq_len=ring_max_seq_len,
            # Captured traces reserve the full history; device metadata bounds valid reads.
            logical_n=ring_logical_n,
            kv_actual_global=chunk_offset,
            sliding_window=sliding_window,
            scale=1.0,
            compute_kernel_config=cp_ring_ckc,
            layer_idx=ring_layer_idx,
            num_layers=ring_num_layers,
        )
    tt_q.deallocate(True)
    if tt_k is not None:
        tt_k.deallocate(True)
    tt_v.deallocate(True)
    tt_out = concat_heads(tt_sdpa)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)
    return tt_out
