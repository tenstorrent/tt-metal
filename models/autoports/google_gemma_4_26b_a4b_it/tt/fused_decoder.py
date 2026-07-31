# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device decoder for ``google/gemma-4-26B-A4B-it``.

This stage intentionally inherits the functional decoder's attention, paged
KV-cache, long-context, and sparse-MoE contracts.  Only numerically equivalent
single-device graph rewrites live here; later program-config, multichip,
full-model, and serving optimization stages are out of scope.
"""

from __future__ import annotations

from typing import Any

import ttnn

from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_Q_HEADS,
    NUM_EXPERTS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
)
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config


class FusedDecoder(FunctionalDecoder):
    """Functional Gemma-4 layer with measured, correctness-preserving fusions."""

    FUSION_PATTERNS = (
        "dense_geglu_activation_folded_into_multiply",
        "serving_decode_sparse_geglu_activation_folded_into_multiply",
        "native_geometry_paged_kv_updates_fused",
        "router_static_scales_folded_into_projection",
    )
    GELU_APPROX = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        **kwargs: Any,
    ) -> "FusedDecoder":
        decoder = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            **kwargs,
        )

        # Functional graph:
        #   rms_norm(x) -> mul(per_hidden_scale) -> mul(H**-0.5)
        #   -> typecast(fp32) -> linear(router_projection)
        # The two operands are immutable.  Pre-fold their product into the
        # FP32 projection once, outside prefill/decode and trace capture.
        scale = ttnn.reshape(decoder.weights.router_scale, [1, 1, HIDDEN_SIZE, 1])
        scale = ttnn.mul(scale, decoder.router_hidden_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        decoder.fused_router_proj = ttnn.mul(
            decoder.weights.router_proj,
            scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scale.deallocate(True)
        return decoder

    def _finish_parallel_ffn(
        self,
        residual: ttnn.Tensor,
        hidden_1: ttnn.Tensor,
        hidden_2: ttnn.Tensor,
    ) -> ttnn.Tensor:
        # The dedicated residual-input RMSNorm was tested and rejected on P300:
        # it was slower at both decode contract batches.  Keep the faster
        # explicit add followed by the already-dedicated RMSNorm.
        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._apply_layer_scalar(hidden_states)

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
        """Run the fused path for one user and any positive logical length."""

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
        hidden_1 = self._rms_norm(self._dense_mlp(mlp_in), self.weights.post_ff_ln_1)

        router_weights = self._router_weights(residual)
        moe_in = self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._rms_norm(self._moe_prefill(moe_in, router_weights), self.weights.post_ff_ln_2)

        hidden_states = self._finish_parallel_ffn(residual, hidden_1, hidden_2)
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
        """Run trace-safe fused paged decode."""

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
        hidden_1 = self._rms_norm(self._dense_mlp(mlp_in), self.weights.post_ff_ln_1)

        router_weights = self._router_weights(residual)
        moe_in = self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._rms_norm(self._moe_decode(moe_in, router_weights), self.weights.post_ff_ln_2)
        return self._finish_parallel_ffn(residual, hidden_1, hidden_2)

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Longer isolated A/B measurements found that the consumer fusion wins
        # for full-attention batch 32 and prefill, while the accepted producer
        # GELU wins for both sliding decode batches and full-attention batch 1.
        tokens = x.shape[-2]
        use_producer_gelu = (self.layer_kind.name == "full_attention" and tokens == 1) or (
            self.layer_kind.name == "sliding_attention" and tokens in (1, 32)
        )
        if use_producer_gelu:
            return FunctionalDecoder._dense_mlp(self, x)

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
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[self.GELU_APPROX],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _disjoint_cache_update_memory_config(self, occupied: ttnn.MemoryConfig, batch: int, width: int):
        """Allocate a batch-height shard disjoint from ``occupied``."""

        grid = self.mesh_device.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        available = all_cores.subtract(occupied.shard_spec.grid)
        start = available.ranges()[0].start
        update_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start, batch, available, True)
        return ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, width),
            core_grid=update_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

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
        """Select the measured cache-update graph for this layer geometry."""

        if self.layer_kind.name == "full_attention" and x.shape[-2] == 32:
            return super()._attention_decode(
                x,
                position_cos=position_cos,
                position_sin=position_sin,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                cache_position_modulo=cache_position_modulo,
            )
        return self._attention_decode_with_fused_cache(
            x,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )

    def _attention_decode_with_fused_cache(
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
        """Decode attention while forcing the eligible fused K/V update."""

        kind = self.layer_kind
        batch = x.shape[-2]
        xqkv = ttnn.linear(x, self.weights.qkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv_head_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            memory_config=qkv_head_mem_config,
        )
        q_mem_config = q_heads.memory_config()
        k_mem_config = k_heads.memory_config()
        v_mem_config = v_heads.memory_config()
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
            rope_mem_config = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            position_cos = ttnn.interleaved_to_sharded(position_cos, rope_mem_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, rope_mem_config)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=True)

        key_cache, value_cache = kv_cache
        native_geometry = (
            key_cache.shape[1] == kind.num_kv_heads
            and key_cache.shape[2] == kind.block_size
            and key_cache.shape[3] == kind.head_dim
        )
        if native_geometry and cache_position_modulo is None:
            v_update_mem_config = self._disjoint_cache_update_memory_config(
                k_heads.memory_config(),
                batch,
                kind.head_dim,
            )
            v_heads = ttnn.to_memory_config(v_heads, v_update_mem_config, dtype=v_heads.dtype)
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
            **self._cache_view_kwargs(prefill=False),
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

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        tokens = residual.shape[-2]
        router_in = self._rms_norm(residual, None)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.fused_router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.typecast(routing, ttnn.bfloat16)
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Sparse experts with GeGLU folded into the multiply consumer."""

        batch = hidden_states.shape[2]
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(batch, MOE_INTERMEDIATE_SIZE)
        down_config = _build_sparse_matmul_config(batch, HIDDEN_SIZE)

        gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_gate,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))

        up = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_up,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
        down_input = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[self.GELU_APPROX],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))

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

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Use sparse GeGLU fusion only where its measured trace wins."""

        if hidden_states.shape[2] == 1:
            return FunctionalDecoder._moe_decode_single_user(self, hidden_states, routing_weights)
        return super()._moe_decode(hidden_states, routing_weights)
