# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe-only batching of shared MLP/reduction/tail; never installed by serving."""

from types import MethodType

import ttnn
from models.demos.gemma4.tt.attention.operations import concat_heads, split_qkv_heads_decode


def attention_decode(
    decoder, hidden_states, *, rope_mats, current_pos, cache_pos, page_table, kv_cache, row_sdpa=False
):
    """Batch dense projections/cache/SDPA, preserving per-row norm and RoPE."""
    attention = decoder.layer.self_attn
    cfg = attention.source.config
    batch = hidden_states.shape[-2]
    assert getattr(decoder, "decode_residual_memory", None) is None
    normed = ttnn.concat(
        [
            decoder.normalize(hidden_states[:, :, slot : slot + 1], cfg.rms_norm_eps, decoder.input_norm_weight)
            for slot in range(batch)
        ],
        dim=2,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    projection = attention.source.weights.wqkv
    assert projection.decode_dram is None
    if getattr(projection, "input_bfp8", False):
        normed = ttnn.typecast(normed, ttnn.bfloat8_b)
    qkv = ttnn.linear(
        normed,
        projection.weight,
        dtype=ttnn.float32,
        compute_kernel_config=projection.decode_compute,
        program_config=projection.program,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    q, k, v = split_qkv_heads_decode(qkv, cfg, attention.source.weights.is_global)
    q, k, v = (ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v))
    cos, sin = (
        ttnn.unsqueeze_to_4D(ttnn.embedding(current_pos, table, layout=ttnn.TILE_LAYOUT)) for table in rope_mats
    )
    qs, ks, vs = [], [], []
    for slot in range(batch):
        qr, kr, vr = (t[:, slot : slot + 1] for t in (q, k, v))
        if attention.common_kv and attention.source.weights.is_global:
            vr = attention.normalize(kr, cfg.rms_norm_eps)
            kr = ttnn.mul(vr, attention.k_weight)
            qr = attention.normalize(qr, cfg.rms_norm_eps, attention.q_weight)
        else:
            qr = attention.normalize(qr, cfg.rms_norm_eps, attention.q_weight)
            kr = attention.normalize(kr, cfg.rms_norm_eps, attention.k_weight)
            vr = attention.normalize(vr, cfg.rms_norm_eps)
        tables = (cos[:, :, slot : slot + 1], sin[:, :, slot : slot + 1])
        qs.append(attention.rotary(qr, *tables, decode=True))
        ks.append(attention.rotary(kr, *tables, decode=True))
        vs.append(vr)
    q, k, v = (ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) for parts in (qs, ks, vs))
    grid_x = min(batch, 8)
    assert batch % grid_x == 0
    grid_y = batch // grid_x
    memories = [
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, start), ttnn.CoreCoord(grid_x - 1, start + grid_y - 1))}
                ),
                (32, cfg.head_dim),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        for start in (0, grid_y)
    ]
    updates = [attention.cache_cast(t, ttnn.bfloat16, mem) for t, mem in zip((k, v), memories)]
    ttnn.experimental.paged_fused_update_cache(
        kv_cache[0], updates[0], kv_cache[1], updates[1], update_idxs_tensor=cache_pos, page_table=page_table
    )
    if row_sdpa:
        attended = ttnn.concat(
            [
                attention.decode_sdpa(
                    q[:, slot : slot + 1],
                    *kv_cache,
                    cur_pos_tensor=cache_pos[slot : slot + 1],
                    page_table_tensor=page_table[slot : slot + 1],
                )
                for slot in range(batch)
            ],
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        attended = attention.decode_sdpa(q, *kv_cache, cur_pos_tensor=cache_pos, page_table_tensor=page_table)
    combined = concat_heads(
        attended, True, num_heads=cfg.num_attention_heads, head_dim=cfg.head_dim, mesh_device=decoder.mesh_device
    )
    combined = ttnn.to_memory_config(combined, ttnn.L1_MEMORY_CONFIG)
    if getattr(attention, "output_input_bfp8", False):
        combined = ttnn.typecast(ttnn.to_memory_config(combined, ttnn.L1_MEMORY_CONFIG), ttnn.bfloat8_b)
    assert getattr(attention, "decode_output_dram", None) is None
    assert getattr(attention, "decode_output_fused", None) is None
    return attention.reduce(
        ttnn.linear(
            combined,
            attention.source.weights.o_proj,
            dtype=ttnn.float32,
            program_config=attention.output_program,
            compute_kernel_config=attention.output_compute,
            memory_config=attention.output_memory,
        )
    )


def shared_decode(shared, x):
    """Use the selected decode weights and arithmetic for every logical row."""
    assert shared.decode_weights is not None
    if getattr(shared, "input_bfp8", False):
        x = ttnn.typecast(x, ttnn.bfloat8_b)
    gu = ttnn.linear(
        x,
        shared.decode_weights[0],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=shared.decode_compute,
        program_config=shared.decode_programs[0],
    )
    up, gate = gu[..., : shared.width], gu[..., shared.width :]
    hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)])
    if getattr(shared, "input_bfp8", False):
        hidden = ttnn.typecast(hidden, ttnn.bfloat8_b)
    return ttnn.linear(
        hidden,
        shared.decode_weights[1],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=shared.decode_compute,
        program_config=shared.decode_programs[1],
    )


def norm_batch(decoder, value, weight=None):
    assert decoder.use_sharded_norms and decoder.sharded_norm_site == "all"
    memory, program = decoder.layer.post_feedforward_layernorm_1._sharded_cfg
    value = ttnn.to_memory_config(ttnn.typecast(value, ttnn.float32), memory)
    result = ttnn.rms_norm(
        value,
        epsilon=decoder.config.rms_norm_eps,
        program_config=program,
        compute_kernel_config=decoder.layer.self_attn.compute,
        memory_config=memory,
    )
    result = ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)
    return result if weight is None else ttnn.mul(result, weight, memory_config=ttnn.L1_MEMORY_CONFIG)


def shared_norm_decode(decoder, hidden_states, *, rope_mats, current_pos, cache_pos, page_table, kv_cache):
    batch = hidden_states.shape[-2]
    normed = norm_batch(decoder, hidden_states, decoder.input_norm_weight)
    # Attention returns a shared persistent all-reduce buffer. Each slot must
    # own a snapshot before the next slot reuses that collective output.
    attentions = [
        ttnn.clone(
            decoder.layer.self_attn(
                normed[:, :, slot : slot + 1],
                rope_mats=rope_mats,
                position_idx=current_pos[:, slot : slot + 1],
                position_idx_cache=cache_pos[slot : slot + 1],
                page_table=page_table[slot : slot + 1],
                kv_cache=kv_cache,
                is_decode=True,
            )
        )
        for slot in range(batch)
    ]
    attention = ttnn.concat(attentions, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    residual = ttnn.add(
        ttnn.typecast(hidden_states, ttnn.float32), norm_batch(decoder, attention, decoder.post_attention_norm_weight)
    )
    normalized = norm_batch(decoder, residual)
    expert_input = ttnn.mul(normalized, decoder.expert_norm_weight, dtype=ttnn.bfloat16)
    routed_rows = []
    for slot in range(batch):
        routes = decoder.layer.moe.router(residual[:, :, slot : slot + 1], normalized=normalized[:, :, slot : slot + 1])
        routed_rows.append(decoder.layer.moe.experts(expert_input[:, :, slot : slot + 1], routes))
    shared = shared_decode(
        decoder.layer.shared_mlp, ttnn.mul(normalized, decoder.shared_norm_weight, dtype=ttnn.bfloat16)
    )
    shared, routed = decoder._reduce_moe_pair(
        shared, ttnn.concat(routed_rows, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    )
    return decoder._fused_tail(residual, shared, routed, True)


def expert_union_decode(experts, x, routes, indices, *, vector_mix=False, indexed_union=False):
    """Selected decode weights/fidelity, union projections, original top8 mix order."""
    assert not experts.expert_split and experts.indexed_router is not None
    batch = x.shape[-2]
    if experts.decode_activation_dtype is not None:
        x = ttnn.typecast(x, experts.decode_activation_dtype)
    if indexed_union:
        # Include a selected expert even if its BF16 routing weight is zero.
        ids = ttnn.concat(indices, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        selected = ttnn.scatter(
            ttnn.zeros_like(routes), dim=-1, index=ids, src=ttnn.ones_like(ids, dtype=ttnn.bfloat16)
        )
    else:
        selected = routes
    sparsity = ttnn.to_layout(ttnn.sum(selected, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
    common = dict(
        sparsity=sparsity,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=ttnn.Tile([32, 32]),
        dtype=ttnn.bfloat16,
        compute_kernel_config=experts.decode_compute,
    )
    gu = ttnn.sparse_matmul(x, experts.gate_up, program_config=experts.gate_config, **common)
    gu = ttnn.reshape(gu, (1, experts.config.num_experts, batch, 2 * experts.width))
    gate, up = gu[..., : experts.width], gu[..., experts.width :]
    hidden = (
        ttnn.mul(gate, up, input_tensor_a_activations=experts.decode_gelu_activations)
        if experts.decode_gelu_activations is not None
        else ttnn.mul(ttnn.gelu(gate, variant=ttnn.GeluVariant.Accurate), up)
    )
    down = ttnn.sparse_matmul(
        hidden, experts.down, program_config=experts.down_config, is_input_a_sparse=True, **common
    )
    down = ttnn.reshape(down, (1, experts.config.num_experts, batch, experts.config.hidden_size))
    down = ttnn.permute(down, (0, 2, 1, 3))
    if vector_mix:
        ids = ttnn.concat(indices, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        mix = ttnn.gather(ttnn.to_layout(routes, ttnn.ROW_MAJOR_LAYOUT), dim=-1, index=ids)
        mix = ttnn.reshape(ttnn.to_layout(mix, ttnn.TILE_LAYOUT), (1, batch, 1, experts.config.top_k))
        gather_ids = ttnn.repeat(
            ttnn.reshape(ttnn.typecast(ids, ttnn.uint32), (1, batch, experts.config.top_k, 1)),
            (1, 1, 1, experts.config.hidden_size),
        )
        chosen = ttnn.gather(ttnn.to_layout(down, ttnn.ROW_MAJOR_LAYOUT), dim=2, index=gather_ids)
        cfg = experts.mix_program
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
            in0_block_w=cfg.in0_block_w,
            out_subblock_h=cfg.out_subblock_h,
            out_subblock_w=cfg.out_subblock_w,
            out_block_h=cfg.out_block_h,
            out_block_w=cfg.out_block_w,
            per_core_M=cfg.per_core_M,
            per_core_N=cfg.per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        output = ttnn.matmul(
            mix,
            ttnn.to_layout(chosen, ttnn.TILE_LAYOUT),
            dtype=ttnn.bfloat16,
            memory_config=experts.mix_memory,
            program_config=program,
            compute_kernel_config=experts.mix_compute,
        )
        return ttnn.reshape(output, (1, 1, batch, experts.config.hidden_size))
    outputs = []
    for slot in range(batch):
        weight = ttnn.reshape(down[:, slot : slot + 1], (experts.config.num_experts, experts.config.hidden_size))
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        chosen = ttnn.embedding(ttnn.typecast(indices[slot], ttnn.uint32), weight, layout=ttnn.TILE_LAYOUT)
        chosen = ttnn.reshape(chosen, (1, 1, experts.config.top_k, experts.config.hidden_size))
        mix_weights = ttnn.gather(
            ttnn.to_layout(routes[:, :, slot : slot + 1], ttnn.ROW_MAJOR_LAYOUT),
            dim=-1,
            index=indices[slot],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        outputs.append(
            ttnn.matmul(
                ttnn.to_layout(mix_weights, ttnn.TILE_LAYOUT),
                chosen,
                dtype=ttnn.bfloat16,
                memory_config=experts.mix_memory,
                program_config=experts.mix_program,
                compute_kernel_config=experts.mix_compute,
            )
        )
    return ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def install_shared_batch(
    model,
    *,
    batch_attention=False,
    batch_norms=False,
    row_sdpa=False,
    batch_experts=False,
    vector_mix=False,
    indexed_union=False,
):
    for decoder in model.layers:
        original = decoder.decode_forward

        def decode(self, hidden_states, *, rope_mats, current_pos, cache_pos, page_table, kv_cache, fallback=original):
            batch = hidden_states.shape[-2]
            if batch == 1:
                return fallback(
                    hidden_states,
                    rope_mats=rope_mats,
                    current_pos=current_pos,
                    cache_pos=cache_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                )
            assert batch <= 32 and not self.sharded_residual and self.grouped_moe_reduce and self.fused_tail
            self._validate_kv_cache(kv_cache)
            if batch_norms:
                return shared_norm_decode(
                    self,
                    hidden_states,
                    rope_mats=rope_mats,
                    current_pos=current_pos,
                    cache_pos=cache_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                )
            eps = self.config.rms_norm_eps
            residuals, routed_rows, shared_inputs = [], [], []
            expert_inputs, route_rows, route_indices = [], [], []
            attention_batch = (
                attention_decode(
                    self,
                    hidden_states,
                    rope_mats=rope_mats,
                    current_pos=current_pos,
                    cache_pos=cache_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    row_sdpa=row_sdpa,
                )
                if batch_attention
                else None
            )
            for slot in range(batch):
                x = hidden_states[:, :, slot : slot + 1, :]
                memory = getattr(self, "decode_residual_memory", None)
                if memory is not None:
                    x = ttnn.to_memory_config(x, memory)
                if attention_batch is None:
                    normed = self.normalize(x, eps, self.input_norm_weight)
                    attention = self.layer.self_attn(
                        normed,
                        rope_mats=rope_mats,
                        position_idx=current_pos[:, slot : slot + 1],
                        position_idx_cache=cache_pos[slot : slot + 1],
                        page_table=page_table[slot : slot + 1, :],
                        kv_cache=kv_cache,
                        is_decode=True,
                    )
                else:
                    attention = attention_batch[:, :, slot : slot + 1]
                residual = ttnn.add(
                    ttnn.typecast(x, ttnn.float32), self.normalize(attention, eps, self.post_attention_norm_weight)
                )
                normalized = self.normalize(residual, eps)
                routes = self.layer.moe.router(residual, normalized=normalized)
                expert_input = ttnn.mul(normalized, self.expert_norm_weight, dtype=ttnn.bfloat16)
                if memory is not None:
                    expert_input = ttnn.to_memory_config(expert_input, ttnn.L1_MEMORY_CONFIG)
                if batch_experts:
                    expert_inputs.append(expert_input)
                    route_rows.append(routes)
                    route_indices.append(ttnn.clone(self.layer.moe.router.decode_indices()))
                else:
                    routed_rows.append(self.layer.moe.experts(expert_input, routes))
                shared_inputs.append(ttnn.mul(normalized, self.shared_norm_weight, dtype=ttnn.bfloat16))
                residuals.append(residual)
            concat = lambda rows: ttnn.concat(rows, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            shared = shared_decode(self.layer.shared_mlp, concat(shared_inputs))
            routed = (
                expert_union_decode(
                    self.layer.moe.experts.decode,
                    concat(expert_inputs),
                    concat(route_rows),
                    route_indices,
                    vector_mix=vector_mix,
                    indexed_union=indexed_union,
                )
                if batch_experts
                else concat(routed_rows)
            )
            shared, routed = self._reduce_moe_pair(shared, routed)
            return self._fused_tail(concat(residuals), shared, routed, True)

        decoder.decode_forward = MethodType(decode, decoder)
