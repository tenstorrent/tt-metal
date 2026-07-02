# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local decode path used only for commit-append.

The generation loop commits a denoised canvas token-by-token into the Gemma4 KV
cache. That path needs a few low-L1 placement choices that are specific to the
DiffusionGemma run-first smoke and should not live in the shared Gemma4 decode
modules. Keep those choices local here and route ``commit_canvas_tokens``
through this file instead of ``Gemma4Model.ttnn_decode_forward``.
"""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    effective_block_size,
    split_qkv_heads_decode,
)
from models.demos.gemma4.tt.attention.weights import AttentionWeights
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.demos.gemma4.tt.experts.weights import ExpertWeights
from models.demos.gemma4.tt.ccl import ccl_allreduce


def _decode_rms_norm_forward(norm, x):
    """Run RMSNorm with the diffusion commit decode fast path when applicable."""
    if len(x.shape) == 4 and x.shape[1] == 1 and 1 <= x.shape[-2] <= ttnn.TILE_SIZE and not x.is_sharded():
        dim = x.shape[-1]
        if norm._sharded_cfg is None or norm._sharded_dim != dim:
            norm._sharded_dim = dim
            norm._sharded_cfg = norm._build_sharded_cfg(dim)
        if norm._sharded_cfg:
            return norm._forward_sharded(x)
    return norm.forward(x)


def _apply_per_head_norm(tensor, weight, eps, with_scale=True):
    """Decode-local per-head norm with the small 8x1 sharded path."""
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]

    def _decode_sharded_norm(flat_tensor):
        tiles = head_dim // ttnn.TILE_SIZE
        if head_dim % ttnn.TILE_SIZE != 0 or tiles % 8 != 0:
            return None

        input_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim // 8),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 1],
            subblock_w=min(tiles // 8, 4),
            block_h=1,
            block_w=tiles // 8,
            inplace=False,
        )
        sharded = ttnn.to_memory_config(flat_tensor, input_memcfg)
        if with_scale and weight is not None:
            normed_sharded = ttnn.rms_norm(sharded, weight=weight, epsilon=eps, program_config=program_config)
        else:
            normed_sharded = ttnn.rms_norm(sharded, epsilon=eps, program_config=program_config)
        sharded.deallocate(True)
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
        normed_sharded.deallocate(True)
        return normed

    if len(orig_shape) == 4 and orig_shape[0] > 1:
        batch, num_heads, seq_len, _ = orig_shape
        flat = ttnn.reshape(tensor, (1, 1, batch * num_heads * seq_len, head_dim))
    else:
        num_heads = orig_shape[1]
        seq_or_batch = orig_shape[2]
        flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_or_batch, head_dim))

    normed = None
    if len(orig_shape) == 4 and orig_shape[0] == 1 and orig_shape[1] == 1 and orig_shape[2] <= ttnn.TILE_SIZE:
        normed = _decode_sharded_norm(flat)
    if normed is None:
        if with_scale and weight is not None:
            normed = ttnn.rms_norm(flat, weight=weight, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            normed = ttnn.rms_norm(flat, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return ttnn.reshape(normed, orig_shape)


def _rotate_half(x):
    hd = x.shape[-1]
    x1 = x[..., : hd // 2]
    x2 = x[..., hd // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _apply_rope_decode_peruser(tensor, cos_b, sin_b):
    heads = tensor.shape[2]
    repeated_cos_sin = False
    if cos_b.shape[2] != heads:
        cos_b = ttnn.repeat(cos_b, ttnn.Shape([1, 1, heads, 1]))
        sin_b = ttnn.repeat(sin_b, ttnn.Shape([1, 1, heads, 1]))
        repeated_cos_sin = True

    scaled = ttnn.mul(tensor, cos_b)
    rotated = _rotate_half(tensor)
    rotated_scaled = ttnn.mul(rotated, sin_b)
    rotated.deallocate(True)
    result = ttnn.add(scaled, rotated_scaled)
    scaled.deallocate(True)
    rotated_scaled.deallocate(True)
    if repeated_cos_sin:
        cos_b.deallocate(True)
        sin_b.deallocate(True)
    return result


def _commit_attention_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    token_index,
    page_table=None,
    ccl_manager=None,
    is_kv_shared=False,
    position_idx_cache=None,
    sequential_kv_write=False,
):
    tp = mesh_config.tp if mesh_config else 1

    xqkv = apply_qkv_projection(hidden_states, weights)
    tt_q, tt_k, tt_v = split_qkv_heads_decode(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

    q_sharded_mem = tt_q.memory_config()
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_q = _apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if is_kv_shared:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = _apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = _apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    use_embedding_rope = len(cos_cache.shape) == 2
    if use_embedding_rope:
        cos_pos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)
        sin_pos = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.unsqueeze_to_4D(cos_pos)
        sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
        batch = tt_q.shape[1]
        cos_b = ttnn.transpose(cos_pos, 1, 2)
        sin_b = ttnn.transpose(sin_pos, 1, 2)
        cos_b = cos_b[:, :batch, :, :]
        sin_b = sin_b[:, :batch, :, :]
        tt_q = _apply_rope_decode_peruser(tt_q, cos_b, sin_b)
        if not is_kv_shared:
            tt_k = _apply_rope_decode_peruser(tt_k, cos_b, sin_b)
    else:
        tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

    cache_pos = position_idx_cache if position_idx_cache is not None else position_idx
    paged_modulo_kwargs = (
        {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo is not None else {}
    )
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if not is_kv_shared:
            tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
            tt_v = ttnn.to_memory_config(tt_v, q_sharded_mem)

            if page_table is not None:
                num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
                eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
                batch = tt_k.shape[1]
                if sequential_kv_write and batch > 1:
                    _shard_shape = list(q_sharded_mem.shard_spec.shape)
                    _one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
                    single_user_mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(_one_core, _shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    k_seq = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
                    v_seq = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
                    nkv, hd = k_seq.shape[2], k_seq.shape[3]
                    for b in range(batch):
                        kb = ttnn.slice(k_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        vb = ttnn.slice(v_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        kb = ttnn.to_memory_config(kb, single_user_mem)
                        vb = ttnn.to_memory_config(vb, single_user_mem)
                        pos_b = ttnn.slice(cache_pos, [b], [b + 1])
                        pt_b = ttnn.slice(page_table, [b, 0], [b + 1, page_table.shape[1]])
                        ttnn.experimental.paged_update_cache(
                            k_cache,
                            kb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        ttnn.experimental.paged_update_cache(
                            v_cache,
                            vb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        for tensor in (kb, vb, pos_b, pt_b):
                            tensor.deallocate(True)
                    k_seq.deallocate(True)
                    v_seq.deallocate(True)
                else:
                    ttnn.experimental.paged_update_cache(
                        k_cache,
                        tt_k,
                        update_idxs_tensor=cache_pos,
                        page_table=page_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache,
                        tt_v,
                        update_idxs_tensor=cache_pos,
                        page_table=page_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
    else:
        k_cache = tt_k
        v_cache = tt_v

    sliding_window = config.sliding_window if config.is_sliding else None
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    if tt_q.memory_config().buffer_type != ttnn.BufferType.DRAM:
        tt_q_l1 = tt_q
        tt_q = ttnn.to_memory_config(tt_q_l1, ttnn.DRAM_MEMORY_CONFIG)
        tt_q_l1.deallocate(True)

    if page_table is not None:
        sdpa_num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        tt_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_pos,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
            block_size=effective_block_size(k_cache, config.head_dim, sdpa_num_local_kv_heads),
            num_kv_heads=sdpa_num_local_kv_heads,
            **paged_modulo_kwargs,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_pos,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    tt_q.deallocate(True)

    num_local_heads = config.num_attention_heads // tp
    tt_out = concat_heads(
        tt_sdpa, is_decode_mode=True, num_heads=num_local_heads, head_dim=config.head_dim, mesh_device=mesh_device
    )
    tt_sdpa.deallocate(True)
    tt_out = apply_output_projection(tt_out, weights)
    return apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)


def _commit_router_forward(router, hidden_states):
    normed = _decode_rms_norm_forward(router.norm, hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)
    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)

    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)
    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    top_k_values = ttnn.div(top_k_values, top_k_sum)
    top_k_sum.deallocate(True)

    dense_routing = ttnn.scatter(
        ttnn.zeros_like(router_probs),
        dim=-1,
        index=top_k_indices,
        src=top_k_values,
    )
    router_probs.deallocate(True)
    top_k_values.deallocate(True)
    top_k_indices.deallocate(True)

    if router.per_expert_scale is not None:
        dense_routing = ttnn.mul(dense_routing, router.per_expert_scale)

    return dense_routing


def _commit_experts_decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    batch_size = hidden_states.shape[2]
    num_experts = config.num_experts
    top_k = config.top_k
    intermediate_size = weights.intermediate_size_per_device

    sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(batch_size, intermediate_size)
    down_config = _build_sparse_matmul_config(batch_size, config.hidden_size)

    gate_sparse = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    sm_intermediate = gate_sparse.shape[-1]
    gate_4d = ttnn.reshape(gate_sparse, (batch_size, num_experts, 1, sm_intermediate))
    gate_sparse.deallocate(True)
    gate_transposed = ttnn.transpose(gate_4d, 1, 2)
    gate_4d.deallocate(True)
    gate = ttnn.reshape(gate_transposed, (batch_size, num_experts, sm_intermediate))
    gate_transposed.deallocate(True)

    up_sparse = ttnn.sparse_matmul(
        hidden_states,
        weights.up_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    up_4d = ttnn.reshape(up_sparse, (batch_size, num_experts, 1, sm_intermediate))
    up_sparse.deallocate(True)
    up_transposed = ttnn.transpose(up_4d, 1, 2)
    up_4d.deallocate(True)
    up = ttnn.reshape(up_transposed, (batch_size, num_experts, sm_intermediate))
    up_transposed.deallocate(True)

    down_input = apply_geglu(gate, up)
    gate.deallocate(True)
    up.deallocate(True)

    down_input_transposed = ttnn.transpose(down_input, 1, 0)
    down_input.deallocate(True)
    down_input = ttnn.reshape(down_input_transposed, (1, num_experts, batch_size, sm_intermediate))
    down_input_transposed.deallocate(True)

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )

    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, num_experts, config.hidden_size))
    routing_3d = ttnn.reshape(routing_weights, (batch_size, num_experts, 1))
    next_states = ttnn.mul(next_states, routing_3d)
    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, config.hidden_size),
        (1, 1, max(32, batch_size), config.hidden_size),
    )

    if mesh_config is not None and mesh_config.tp > 1:
        next_states = ccl_allreduce(next_states, mesh_config, ccl_manager)

    return next_states


def _commit_moe_forward(moe, router_input, expert_input):
    dense_routing = _commit_router_forward(moe.router, router_input)
    output = _commit_experts_decode_forward(
        expert_input,
        dense_routing,
        weights=moe.experts.weights,
        config=moe.experts.config,
        mesh_config=moe.experts.mesh_config,
        mesh_device=moe.experts.mesh_device,
        ccl_manager=moe.experts.ccl_manager,
    )
    dense_routing.deallocate(True)
    return output


def _commit_layer_forward(
    layer,
    hidden_states,
    rope_mats,
    position_idx,
    page_table,
    kv_cache,
    token_index=None,
    per_layer_input=None,
    is_kv_shared=False,
    position_idx_cache=None,
):
    residual = hidden_states
    normed = _decode_rms_norm_forward(layer.input_layernorm, hidden_states)
    attn_output = _commit_attention_decode_forward(
        hidden_states=normed,
        cos_cache=rope_mats[0],
        sin_cache=rope_mats[1],
        weights=layer.self_attn.weights,
        kv_cache=kv_cache or layer.self_attn.kv_cache,
        config=layer.self_attn.config,
        mesh_config=layer.self_attn.mesh_config,
        mesh_device=layer.self_attn.mesh_device,
        position_idx=position_idx,
        token_index=token_index,
        page_table=page_table,
        ccl_manager=layer.self_attn.ccl_manager,
        is_kv_shared=is_kv_shared,
        position_idx_cache=position_idx_cache,
    )
    attn_output = _decode_rms_norm_forward(layer.post_attention_layernorm, attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = _decode_rms_norm_forward(layer.pre_feedforward_layernorm, hidden_states)
    mlp_output = layer.shared_mlp(normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = _decode_rms_norm_forward(layer.post_feedforward_layernorm_1, mlp_output)
        mlp_output.deallocate(True)

        residual_for_router = residual
        expert_input = _decode_rms_norm_forward(layer.pre_feedforward_layernorm_2, residual_for_router)
        expert_output = _commit_moe_forward(layer.moe, residual_for_router, expert_input)
        expert_input.deallocate(True)

        expert_normed = _decode_rms_norm_forward(layer.post_feedforward_layernorm_2, expert_output)
        expert_output.deallocate(True)

        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = _decode_rms_norm_forward(layer.post_feedforward_layernorm, hidden_states)
    combined = ttnn.add(residual, hidden_states)
    residual.deallocate(True)
    hidden_states.deallocate(True)
    hidden_states = combined

    if layer.hidden_size_per_layer_input and per_layer_input is not None and hasattr(layer, "per_layer_input_gate"):
        residual_pli = hidden_states
        gated = ttnn.linear(hidden_states, layer.per_layer_input_gate)
        gated = ttnn.gelu(gated, fast_and_approximate_mode=True)
        gated = ttnn.mul(gated, per_layer_input)
        projected = ttnn.linear(gated, layer.per_layer_projection)
        normed_pli = _decode_rms_norm_forward(layer.post_per_layer_input_norm, projected)
        hidden_states = ttnn.add(residual_pli, normed_pli)
        if len(hidden_states.shape) > 4:
            hidden_states = ttnn.reshape(hidden_states, (1, 1, hidden_states.shape[-2], layer.hidden_size))

    if layer.layer_scalar != 1.0:
        hidden_states = ttnn.mul(hidden_states, layer.layer_scalar)

    return hidden_states


def _commit_model_forward(
    tt_model,
    hidden_states,
    *,
    position_idx,
    page_table=None,
    kv_caches=None,
    token_index=None,
    position_idx_cache=None,
    pli_combined=None,
    page_tables_per_layer=None,
):
    caches = kv_caches or tt_model.tt_kv_cache
    if page_tables_per_layer is not None and len(page_tables_per_layer) != len(tt_model.layers):
        raise ValueError(
            f"page_tables_per_layer has {len(page_tables_per_layer)} entries "
            f"but model has {len(tt_model.layers)} layers"
        )

    for i, layer in enumerate(tt_model.layers):
        layer_rope = tt_model._get_rope_mats(i, for_decode=True)
        layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table
        pli_tt = pli_combined[:, :, i : i + 1, :] if pli_combined is not None else None
        hidden_states = _commit_layer_forward(
            layer,
            hidden_states,
            rope_mats=layer_rope,
            position_idx=position_idx,
            page_table=layer_page_table,
            kv_cache=caches[i] if caches else None,
            token_index=token_index,
            per_layer_input=pli_tt,
            is_kv_shared=i in tt_model.kv_shared_layer_map,
            position_idx_cache=position_idx_cache,
        )

    hidden_states = _decode_rms_norm_forward(tt_model.norm, hidden_states)
    return tt_model._apply_lm_head(hidden_states, is_decode=True)


def commit_decode_forward(
    tt_model,
    x,
    current_pos,
    rot_mat_idxs=None,
    page_table=None,
    kv_cache=None,
    on_device_logits=False,
    pli_combined=None,
    page_tables_per_layer=None,
):
    """Run one DiffusionGemma commit-append decode step."""
    if x.dtype in (ttnn.uint32, ttnn.int32):
        input_embeds = tt_model.embed_tokens(x)
        if len(input_embeds.shape) == 3:
            input_embeds = ttnn.unsqueeze_to_4D(input_embeds)
        input_embeds = ttnn.to_layout(input_embeds, ttnn.TILE_LAYOUT)
    else:
        input_embeds = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    token_index = None if tt_model.rope_caches_2d else 0
    position_idx_cache = rot_mat_idxs

    if page_tables_per_layer is None:
        page_tables_per_layer = getattr(tt_model, "_active_page_tables_per_layer", None)
    page_tables_per_layer = tt_model._page_tables_to_ttnn(page_tables_per_layer)

    if pli_combined is None:
        pli_combined = getattr(tt_model, "_decode_pli_combined", None)
    pli_combined = ttnn.to_layout(pli_combined, ttnn.TILE_LAYOUT) if pli_combined is not None else None

    logits = _commit_model_forward(
        tt_model,
        input_embeds,
        position_idx=current_pos,
        page_table=page_table,
        kv_caches=kv_cache,
        token_index=token_index,
        position_idx_cache=position_idx_cache,
        pli_combined=pli_combined,
        page_tables_per_layer=page_tables_per_layer,
    )

    if on_device_logits:
        assert tt_model.sampling is not None, (
            "commit_decode_forward got on_device_logits=True but no on-device sampling module exists "
            "(tt_model.sampling is None)."
        )
        batch_dim = logits.shape[2]
        if batch_dim < 32:
            logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, 32 - batch_dim), (0, 0)], value=0.0)
        return logits

    return logits, None
