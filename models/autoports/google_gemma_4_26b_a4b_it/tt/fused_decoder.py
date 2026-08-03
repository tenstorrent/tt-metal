# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device decoder for ``google/gemma-4-26B-A4B-it``.

The functional decoder remains the semantic source of truth for tensor,
paged-cache, and trace contracts.  This class inherits setup and unchanged
dedicated-op paths, but owns the measured prefill/decode bodies and the fused
subgraphs.  It never dispatches a measured call back to a functional forward
method.
"""

from __future__ import annotations

from typing import Any

import ttnn

from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    NUM_Q_HEADS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _build_sparse_matmul_config,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
)


class FusedDecoder(FunctionalDecoder):
    """Functional-equivalent decoder with device graph fusions enabled."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The HF checkpoint already stores gate/up together.  The functional
        # decoder separates them for clarity; Stage 02 repacks them once so a
        # single sparse projection can feed both GeGLU operands.
        self.expert_up_gate = ttnn.concat(
            [self.weights.expert_up, self.weights.expert_gate],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Every fused MoE entry point uses the packed tensor.  Release the two
        # source buffers after the device concat so packing does not reduce the
        # decoder's advertised KV-cache capacity.
        self.weights.expert_up.deallocate(True)
        self.weights.expert_gate.deallocate(True)

    def _merge_ffn_branches(self, hidden_1: ttnn.Tensor, hidden_2: ttnn.Tensor) -> ttnn.Tensor:
        """Merge the two FFN branches and normalize their sum."""
        merged = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._rms_norm(merged, self.weights.post_ff_ln)

    def _dense_mlp(self, x: ttnn.Tensor, *, fold_activation: bool) -> ttnn.Tensor:
        """Use explicit GELU: the folded candidate has no material prefill win."""
        gate = ttnn.linear(
            x,
            self.weights.mlp_gate,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.weights.mlp_up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _packed_expert_activation(self, up_gate: ttnn.Tensor, *, use_composite: bool) -> ttnn.Tensor:
        """Apply the mode/layer-specific winning GeGLU lowering."""
        if use_composite:
            original_shape = tuple(up_gate.shape)
            if len(original_shape) == 3:
                up_gate = ttnn.unsqueeze_to_4D(up_gate)
            output = ttnn.geglu(up_gate, dim=-1, memory_config=up_gate.memory_config())
            if len(original_shape) == 3:
                output = ttnn.reshape(output, (*original_shape[:-1], MOE_INTERMEDIATE_SIZE))
            return output
        rank = len(up_gate.shape)
        starts = [0] * rank
        up_ends = list(up_gate.shape)
        up_ends[-1] = MOE_INTERMEDIATE_SIZE
        gate_starts = [0] * rank
        gate_starts[-1] = MOE_INTERMEDIATE_SIZE
        gate_ends = list(up_gate.shape)
        up = ttnn.slice(up_gate, starts, up_ends)
        gate = ttnn.slice(up_gate, gate_starts, gate_ends)
        return ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
        )

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Project sparse expert up/gate values together and fold GeLU into mul."""
        batch = hidden_states.shape[2]
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        up_gate_config = _build_sparse_matmul_config(batch, 2 * MOE_INTERMEDIATE_SIZE)
        down_config = _build_sparse_matmul_config(batch, HIDDEN_SIZE)

        up_gate = ttnn.sparse_matmul(
            hidden_states,
            self.expert_up_gate,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=up_gate_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        packed_intermediate = up_gate.shape[-1]
        up_gate = ttnn.reshape(up_gate, (batch, NUM_EXPERTS, 1, packed_intermediate))
        up_gate = ttnn.transpose(up_gate, 1, 2)
        up_gate = ttnn.reshape(up_gate, (batch, NUM_EXPERTS, packed_intermediate))
        down_input = self._packed_expert_activation(
            up_gate,
            use_composite=False,
        )
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, MOE_INTERMEDIATE_SIZE))

        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Apply the same packed expert projection to a bounded prefill chunk."""
        seq_len = hidden_states.shape[2]
        if seq_len == TILE_SIZE:
            return self._moe_prefill_tile(hidden_states, routing_weights)
        outputs = []
        for start in range(0, seq_len, TILE_SIZE):
            end = start + TILE_SIZE
            hidden_tile = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, HIDDEN_SIZE])
            routing_tile = ttnn.slice(routing_weights, [0, 0, start, 0], [1, 1, end, NUM_EXPERTS])
            outputs.append(self._moe_prefill_tile(hidden_tile, routing_tile))
        return ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_prefill_tile(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        chunk_len = hidden_states.shape[2]
        group_size = chunk_len // TILE_SIZE
        hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, HIDDEN_SIZE))
        sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, group_size, 1))
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        up_gate_config = _build_sparse_matmul_config(TILE_SIZE, 2 * MOE_INTERMEDIATE_SIZE)
        down_config = _build_sparse_matmul_config(TILE_SIZE, HIDDEN_SIZE)

        up_gate = ttnn.sparse_matmul(
            hidden_grouped,
            self.expert_up_gate,
            sparsity=sparsity,
            nnz=NUM_EXPERTS * group_size,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=up_gate_config,
            dtype=self.activation_dtype,
        )
        hidden_grouped.deallocate(True)
        packed_intermediate = up_gate.shape[-1]
        up_gate = ttnn.transpose(up_gate, 1, 3)
        up_gate = ttnn.reshape(up_gate, (1, NUM_EXPERTS, chunk_len, packed_intermediate))
        down_input = self._packed_expert_activation(up_gate, use_composite=False)
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=self.expert_prefill_sparsity,
            nnz=NUM_EXPERTS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
        )
        down_input.deallocate(True)
        next_states = ttnn.reshape(down, (1, NUM_EXPERTS, chunk_len, HIDDEN_SIZE))
        routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
        next_states = ttnn.mul(next_states, routing_permuted)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        return ttnn.reshape(next_states, (1, 1, chunk_len, HIDDEN_SIZE))

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        """Decode attention with parallel K/V cache updates when expressible."""
        kind = self.layer_kind
        batch = x.shape[-2]
        xqkv = ttnn.linear(x, self.weights.qkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            memory_config=head_mem_config,
        )
        q_mem_config = q_heads.memory_config()
        k_mem_config = k_heads.memory_config()
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        v_update_mem_config = _make_disjoint_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_update_mem_config, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_update_mem_config, dtype=v_heads.dtype)
            rope_mem_config = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            position_cos = ttnn.interleaved_to_sharded(position_cos, rope_mem_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, rope_mem_config)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=True)

        key_cache, value_cache = kv_cache
        natural_cache_view = (
            key_cache.shape[1] == kind.num_kv_heads
            and key_cache.shape[2] == kind.block_size
            and key_cache.shape[3] == kind.head_dim
        )
        if cache_position_modulo is None and natural_cache_view:
            ttnn.experimental.paged_fused_update_cache(
                key_cache,
                k_heads,
                value_cache,
                v_heads,
                update_idxs_tensor=current_pos,
                page_table=page_table,
            )
        else:
            update_kwargs = self._cache_view_kwargs(prefill=False)
            if cache_position_modulo is not None:
                update_kwargs["cache_position_modulo"] = cache_position_modulo
            ttnn.experimental.paged_update_cache(
                key_cache,
                k_heads,
                update_idxs_tensor=current_pos,
                page_table=page_table,
                **update_kwargs,
            )
            ttnn.experimental.paged_update_cache(
                value_cache,
                v_heads,
                update_idxs_tensor=current_pos,
                page_table=page_table,
                **update_kwargs,
            )

        sdpa_kwargs = self._cache_view_kwargs(prefill=False)
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **sdpa_kwargs,
        )
        concat_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        attn_out = ttnn.to_memory_config(attn_out, concat_mem_config, dtype=attn_out.dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
        attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
        attn_out = ttnn.linear(
            attn_out,
            self.weights.o_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if attn_out.shape[-2] != batch:
            attn_out = ttnn.slice(
                attn_out,
                starts=[0, 0, 0, 0],
                ends=[1, 1, batch, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return attn_out

    def _prefill_forward_single_user(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        chunk_page_table: ttnn.Tensor | None,
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        logical_seq_len = hidden_states.shape[-2]
        if logical_seq_len < 1:
            raise ValueError("prefill requires at least one logical token")
        padded_seq_len = ((logical_seq_len + 31) // 32) * 32
        if padded_seq_len != logical_seq_len:
            pad = [(0, 0), (0, 0), (0, padded_seq_len - logical_seq_len), (0, 0)]
            hidden_states = ttnn.pad(hidden_states, pad, 0.0)
            position_cos = ttnn.pad(position_cos, pad, 0.0)
            position_sin = ttnn.pad(position_sin, pad, 0.0)

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_prefill(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            cache_position_modulo=cache_position_modulo,
            logical_seq_len=logical_seq_len,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        mlp_in = self._rms_norm(hidden_states, self.weights.pre_ff_ln)
        mlp_out = self._dense_mlp(mlp_in, fold_activation=True)
        hidden_1 = self._rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = self._router_weights(residual)
        moe_in = self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._moe_prefill(moe_in, router_weights)
        hidden_2 = self._rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = self._merge_ffn_branches(hidden_1, hidden_2)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._apply_layer_scalar(hidden_states)
        if padded_seq_len != logical_seq_len:
            hidden_states = ttnn.slice(
                hidden_states,
                starts=[0, 0, 0, 0],
                ends=[1, 1, logical_seq_len, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return hidden_states

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        if hidden_states.shape[-2] < 1:
            raise ValueError("decode requires at least one batch row")

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        mlp_in = self._rms_norm(hidden_states, self.weights.pre_ff_ln)
        mlp_out = self._dense_mlp(mlp_in, fold_activation=False)
        hidden_1 = self._rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = self._router_weights(residual)
        moe_in = self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._moe_decode(moe_in, router_weights)
        hidden_2 = self._rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = self._merge_ffn_branches(hidden_1, hidden_2)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._apply_layer_scalar(hidden_states)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs: Any) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


__all__ = ["FusedDecoder"]


def _make_disjoint_decode_height_sharded_memory_config(device: Any, batch: int, width: int) -> ttnn.MemoryConfig:
    """Place V updates on a grid disjoint from K for the dual-cache op."""
    grid = device.compute_with_storage_grid_size()
    num_cores = max(1, batch)
    grid_x = min(num_cores, grid.x)
    while num_cores % grid_x != 0 or num_cores // grid_x > grid.y:
        grid_x -= 1
    grid_y = num_cores // grid_x
    if 2 * grid_y <= grid.y:
        start = ttnn.CoreCoord(0, grid_y)
        end = ttnn.CoreCoord(grid_x - 1, 2 * grid_y - 1)
    elif 2 * grid_x <= grid.x:
        start = ttnn.CoreCoord(grid_x, 0)
        end = ttnn.CoreCoord(2 * grid_x - 1, grid_y - 1)
    else:
        raise ValueError(f"cannot place two disjoint {grid_x}x{grid_y} decode grids on {grid.x}x{grid.y}")
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(start, end)])
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
