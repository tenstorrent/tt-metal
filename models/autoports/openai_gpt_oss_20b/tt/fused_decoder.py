# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device GPT-OSS decoder layer.

This stage keeps the functional decoder's BF16/DRAM contract while replacing
spelled-out subgraphs with dedicated TTNN operations and removing allocation-
and layout-only work from the measured prefill and decode paths.
"""

from __future__ import annotations

import math

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import FunctionalDecoder


def _sparse_program_config(output_width: int):
    """Return a rectangular sparse-matmul grid with no idle receivers."""

    core_x, core_y = 5, 6
    core_count = core_x * core_y
    per_core_n = math.ceil(math.ceil(output_width / ttnn.TILE_SIZE) / core_count)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class FusedDecoder(FunctionalDecoder):
    """FunctionalDecoder-compatible layer whose forwards execute the fused graph."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Regular SDPA applies ``scale`` to the attention-sink logit along
        # with QK.  The GPT-OSS reference sink is already in score units, so
        # pre-divide it once at construction and let the kernel rescale it.
        prefill_sink = ttnn.slice(
            self.attention_sinks,
            [0, 0, 0, 0],
            [1, self.num_heads, 1, 1],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prefill_sdpa_sink = ttnn.multiply(
            prefill_sink,
            1.0 / self.scale,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Populate fixed-capacity metadata views on first use. Warmed prefill
        # and traced decode then reuse stable device tensors without dispatch,
        # while unused logical lengths consume no device buffers.
        self.prefill_rotary_views = {}
        self.decode_position_views = {}
        self.moe_policy = "auto"

        # Candidate graph rewrite: split the emitted interleaved gate/up
        # projection once so the runtime graph can avoid two strided slices.
        self.gate_weight = ttnn.slice(
            self.gate_up_weight,
            [0, 0, 0],
            [self.num_experts, self.hidden_size, 2 * self.intermediate_size],
            [1, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_weight = ttnn.slice(
            self.gate_up_weight,
            [0, 0, 1],
            [self.num_experts, self.hidden_size, 2 * self.intermediate_size],
            [1, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gate_bias = ttnn.slice(
            self.gate_up_bias,
            [0, 0, 0],
            [self.num_experts, 1, 2 * self.intermediate_size],
            [1, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_bias = ttnn.slice(
            self.gate_up_bias,
            [0, 0, 1],
            [self.num_experts, 1, 2 * self.intermediate_size],
            [1, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _get_prefill_rotary_views(self, seq_len: int):
        if seq_len not in self.prefill_rotary_views:
            self.prefill_rotary_views[seq_len] = (
                ttnn.slice(
                    self.rotary_cos,
                    [0, 0, 0, 0],
                    [1, 1, seq_len, self.head_dim],
                    [1, 1, 1, 1],
                ),
                ttnn.slice(
                    self.rotary_sin,
                    [0, 0, 0, 0],
                    [1, 1, seq_len, self.head_dim],
                    [1, 1, 1, 1],
                ),
            )
        return self.prefill_rotary_views[seq_len]

    def _get_decode_position_views(self, position: int):
        if position not in self.decode_position_views:
            cos = ttnn.slice(
                self.rotary_cos,
                [0, 0, position, 0],
                [1, 1, position + 1, self.head_dim],
                [1, 1, 1, 1],
            )
            sin = ttnn.slice(
                self.rotary_sin,
                [0, 0, position, 0],
                [1, 1, position + 1, self.head_dim],
                [1, 1, 1, 1],
            )
            update_indices = ttnn.slice(self.position_indices, [position], [position + 1], [1])
            if self.batch > 1:
                update_indices = ttnn.repeat(
                    update_indices,
                    ttnn.Shape([self.batch]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            self.decode_position_views[position] = (cos, sin, update_indices)
        return self.decode_position_views[position]

    def _moe_forward(self, hidden_states, seq_len: int):
        """Run the dense functional MoE with adjacent ops merged in-place."""

        if self.moe_policy in ("sparse", "sparse_split"):
            split_sparse_projection = self.moe_policy == "sparse_split"
            return self._sparse_moe_forward(hidden_states, seq_len, split_projection=split_sparse_projection)

        tokens = self.batch * seq_len
        token_states = ttnn.reshape(hidden_states, [tokens, self.hidden_size])
        expert_input = ttnn.reshape(token_states, [1, tokens, self.hidden_size])
        expert_input = ttnn.repeat(
            expert_input,
            ttnn.Shape([self.num_experts, 1, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        use_split_projection = self.moe_policy == "split" or (self.moe_policy == "auto" and seq_len > 1)
        if use_split_projection:
            # Two narrower projections avoid the expensive strided extraction
            # at prefill sizes and were faster end-to-end on Blackhole.
            gate = ttnn.linear(
                expert_input,
                self.gate_weight,
                bias=self.gate_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            up = ttnn.linear(
                expert_input,
                self.up_weight,
                bias=self.up_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
        else:
            # One wider projection plus extraction is materially faster for a
            # single decode token, despite its extra data-movement kernels.
            gate_up = ttnn.linear(
                expert_input,
                self.gate_up_weight,
                bias=self.gate_up_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 1],
                [self.num_experts, tokens, 2 * self.intermediate_size],
                [1, 1, 2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0],
                [self.num_experts, tokens, 2 * self.intermediate_size],
                [1, 1, 2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        up = ttnn.clamp(
            up,
            -self.swiglu_limit,
            self.swiglu_limit,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=up,
        )
        up = ttnn.add(
            up,
            1.0,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=up,
        )

        gate = ttnn.clamp(
            gate,
            float("-inf"),
            self.swiglu_limit,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=gate,
        )
        # Fold sigmoid into the scalar multiply which produces its input.
        sigmoid = ttnn.multiply(
            gate,
            1.703125,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activations=[ttnn.UnaryOpType.SIGMOID],
        )
        gated = ttnn.multiply(
            gate,
            sigmoid,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=gate,
        )
        down_input = ttnn.multiply(
            up,
            gated,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=up,
        )
        expert_output = ttnn.linear(
            down_input,
            self.down_weight,
            bias=self.down_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        router_input = ttnn.typecast(token_states, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_logits = ttnn.linear(
            router_input,
            self.router_weight,
            bias=self.router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        top_values, top_indices = ttnn.topk(
            router_logits,
            self.experts_per_token,
            1,
            True,
            True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_weights = ttnn.softmax(top_values, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG, numeric_stable=True)
        routing = ttnn.scatter(
            input=ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_weights,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.permute(routing, (1, 0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.reshape(routing, [self.num_experts, tokens, 1])
        weighted = ttnn.multiply(
            expert_output,
            routing,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=expert_output,
        )
        reduced = ttnn.sum(weighted, [0], False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(reduced, [1, self.batch, seq_len, self.hidden_size])

    def _sparse_moe_forward(self, hidden_states, seq_len: int, *, split_projection: bool):
        """Use top-k routing as the sparsity mask for exact BF16 expert matmuls."""

        tokens = self.batch * seq_len
        token_states = ttnn.reshape(hidden_states, [tokens, self.hidden_size])

        router_input = ttnn.typecast(token_states, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_logits = ttnn.linear(
            router_input,
            self.router_weight,
            bias=self.router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        top_values, top_indices = ttnn.topk(
            router_logits,
            self.experts_per_token,
            1,
            True,
            True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_weights = ttnn.softmax(top_values, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG, numeric_stable=True)
        routing = ttnn.scatter(
            input=ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_weights,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sparsity = ttnn.reshape(routing, [1, tokens, 1, self.num_experts])
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)

        expert_input = ttnn.reshape(token_states, [1, tokens, 1, self.hidden_size])
        if split_projection:
            gate_weight = ttnn.reshape(
                self.gate_weight,
                [1, self.num_experts, self.hidden_size, self.intermediate_size],
            )
            up_weight = ttnn.reshape(
                self.up_weight,
                [1, self.num_experts, self.hidden_size, self.intermediate_size],
            )
            gate = ttnn.sparse_matmul(
                expert_input,
                gate_weight,
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                output_tile=ttnn.Tile([32, 32]),
                program_config=_sparse_program_config(self.intermediate_size),
                compute_kernel_config=self.compute_kernel_config,
            )
            up = ttnn.sparse_matmul(
                expert_input,
                up_weight,
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                output_tile=ttnn.Tile([32, 32]),
                program_config=_sparse_program_config(self.intermediate_size),
                compute_kernel_config=self.compute_kernel_config,
            )
            gate = ttnn.reshape(gate, [tokens, self.num_experts, self.intermediate_size])
            gate = ttnn.permute(gate, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate = ttnn.add(
                gate,
                self.gate_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=gate,
            )
            up = ttnn.reshape(up, [tokens, self.num_experts, self.intermediate_size])
            up = ttnn.permute(up, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up = ttnn.add(
                up,
                self.up_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=up,
            )
        else:
            gate_up_weight = ttnn.reshape(
                self.gate_up_weight,
                [1, self.num_experts, self.hidden_size, 2 * self.intermediate_size],
            )
            gate_up = ttnn.sparse_matmul(
                expert_input,
                gate_up_weight,
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                output_tile=ttnn.Tile([32, 32]),
                program_config=_sparse_program_config(2 * self.intermediate_size),
                compute_kernel_config=self.compute_kernel_config,
            )
            gate_up = ttnn.reshape(gate_up, [tokens, self.num_experts, 2 * self.intermediate_size])
            gate_up = ttnn.permute(gate_up, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_up = ttnn.add(
                gate_up,
                self.gate_up_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=gate_up,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 1],
                [self.num_experts, tokens, 2 * self.intermediate_size],
                [1, 1, 2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0],
                [self.num_experts, tokens, 2 * self.intermediate_size],
                [1, 1, 2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        up = ttnn.clamp(
            up,
            -self.swiglu_limit,
            self.swiglu_limit,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=up,
        )
        up = ttnn.add(up, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=up)
        gate = ttnn.clamp(
            gate,
            float("-inf"),
            self.swiglu_limit,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=gate,
        )
        sigmoid = ttnn.multiply(
            gate,
            1.703125,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activations=[ttnn.UnaryOpType.SIGMOID],
        )
        gate = ttnn.multiply(gate, sigmoid, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=gate)
        down_input = ttnn.multiply(up, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=up)

        down_input = ttnn.permute(down_input, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_input = ttnn.reshape(down_input, [tokens, self.num_experts, 1, self.intermediate_size])
        down_weight = ttnn.reshape(
            self.down_weight,
            [1, self.num_experts, self.intermediate_size, self.hidden_size],
        )
        down_sparsity = ttnn.reshape(sparsity, [1, 1, tokens, self.num_experts])
        expert_output = ttnn.sparse_matmul(
            down_input,
            down_weight,
            sparsity=down_sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_program_config(self.hidden_size),
            compute_kernel_config=self.compute_kernel_config,
        )
        expert_output = ttnn.reshape(expert_output, [tokens, self.num_experts, self.hidden_size])
        expert_output = ttnn.permute(expert_output, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_output = ttnn.add(
            expert_output,
            self.down_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=expert_output,
        )
        routing = ttnn.permute(routing, (1, 0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.reshape(routing, [self.num_experts, tokens, 1])
        expert_output = ttnn.multiply(
            expert_output,
            routing,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=expert_output,
        )
        reduced = ttnn.sum(expert_output, [0], False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(reduced, [1, self.batch, seq_len, self.hidden_size])

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        """Run sink-aware fused SDPA for any valid logical prefill length."""

        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1; use decode_forward for one token")
        self._validate_caches(key_cache, value_cache)

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = self._get_prefill_rotary_views(seq_len)
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Prefill rotary_embedding exposes the tile-padded sequence as its
        # logical extent; restore the public non-aligned logical length.
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.batch == 1:
            ttnn.fill_cache(key_cache, key, 0)
            ttnn.fill_cache(value_cache, value, 0)
        else:
            for user_id in range(self.batch):
                user_key = ttnn.slice(
                    key,
                    [user_id, 0, 0, 0],
                    [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                    [1, 1, 1, 1],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                user_value = ttnn.slice(
                    value,
                    [user_id, 0, 0, 0],
                    [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                    [1, 1, 1, 1],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.fill_cache(key_cache, user_key, user_id)
                ttnn.fill_cache(value_cache, user_value, user_id)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            attention_sink=self.prefill_sdpa_sink,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.num_heads * self.head_dim])
        attention = ttnn.linear(
            attention,
            self.output_weight,
            bias=self.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=attention,
        )

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = self._moe_forward(hidden_states, seq_len)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=hidden_states,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        """Run paged decode with indexed RoPE and collapsed concat-head views."""

        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [1, 1, self.batch, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_heads_mem_config,
        )

        # Use construction-time position views. Passing a changing scalar
        # token index to rotary_embedding can reuse the first cached program's
        # value; fixed one-row views plus index zero remain trace-safe.
        cos, sin, update_indices = self._get_decode_position_views(current_pos)
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            0,
            memory_config=self.decode_heads_mem_config,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=self.decode_heads_mem_config,
        )

        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )

        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            attn_mask=None,
            cur_pos_tensor=update_indices,
            attention_sink=self.decode_attention_sinks,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # SDPA decode rejects sharded output for GQA, so this one memory-layout
        # conversion is required by the height-sharded concat-heads kernel.
        attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        # Drop the concat kernel's batch padding, then express the functional
        # graph's singleton-dimension permutation as a reshape view.
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.num_heads * self.head_dim],
            [1, 1, 1, 1],
        )
        attention = ttnn.reshape(
            attention,
            [1, self.batch, 1, self.num_heads * self.head_dim],
        )
        attention = ttnn.linear(
            attention,
            self.output_weight,
            bias=self.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=attention,
        )

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = self._moe_forward(hidden_states, 1)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=hidden_states,
        )
