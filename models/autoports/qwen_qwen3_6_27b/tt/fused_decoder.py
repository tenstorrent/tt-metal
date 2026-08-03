# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device Qwen3.6-27B decoder layer.

This stage deliberately inherits the functional decoder's setup, cache, RoPE,
prefill, decode, and public shape contracts.  Runtime overrides are limited to
rewrites that have a dedicated TTNN fused implementation.
"""

from __future__ import annotations

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder


class FusedDecoder(FunctionalDecoder):
    """Functional decoder with graph-level TTNN fusion enabled."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if decoder.layer_kind == "full_attention":
            decoder.weights["packed_qkv"] = ttnn.concat(
                [decoder.weights["q_proj"], decoder.weights["k_proj"], decoder.weights["v_proj"]],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            decoder.weights["packed_linear_inputs"] = ttnn.concat(
                [
                    decoder.weights["in_qkv"],
                    decoder.weights["in_z"],
                    decoder.weights["in_b"],
                    decoder.weights["in_a"],
                ],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return decoder

    def _mlp(self, hidden_states):
        gate = ttnn.linear(hidden_states, self.weights["mlp_gate"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.linear(hidden_states, self.weights["mlp_up"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(hidden_states, self.weights["mlp_down"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _full_attention_prefill(self, hidden_states, page_table, current_positions):
        projected = ttnn.linear(hidden_states, self.weights["packed_qkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        q_and_gate = projected[..., : 2 * q_width]
        q = q_and_gate[..., :q_width]
        gate = q_and_gate[..., q_width:]
        k = projected[..., 2 * q_width : 2 * q_width + kv_width]
        v = projected[..., 2 * q_width + kv_width :]

        sequence = hidden_states.shape[2]
        q = ttnn.reshape(q, (self.batch, sequence, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (self.batch, sequence, self.num_kv_heads, self.head_dim))
        v = ttnn.reshape(v, (self.batch, sequence, self.num_kv_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        q = self._per_head_norm_prefill(q, "q_norm")
        k = self._per_head_norm_prefill(k, "k_norm")
        q = self._partial_rope_prefill(q, current_positions)
        k = self._partial_rope_prefill(k, current_positions)

        ttnn.experimental.paged_fill_cache(
            self.caches["key"], k, page_table, batch_idx_tensor=self.caches["batch_indices"]
        )
        ttnn.experimental.paged_fill_cache(
            self.caches["value"], v, page_table, batch_idx_tensor=self.caches["batch_indices"]
        )
        if sequence <= 32768:
            attention = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.head_dim**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            chunks = []
            start = 0
            while start < sequence:
                logical_chunk = min(32768, sequence - start)
                q_chunk = ttnn.slice(
                    q,
                    (0, 0, start, 0),
                    (self.batch, self.num_heads, start + logical_chunk, self.head_dim),
                )
                padding = (-logical_chunk) % 32
                if padding:
                    q_chunk = ttnn.pad(
                        q_chunk,
                        ((0, 0), (0, 0), (0, padding), (0, 0)),
                        value=0.0,
                    )
                chunk = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_chunk,
                    self.caches["key"],
                    self.caches["value"],
                    page_table,
                    chunk_start_idx=start,
                    scale=self.head_dim**-0.5,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if padding:
                    chunk = ttnn.slice(
                        chunk,
                        (0, 0, 0, 0),
                        (self.batch, self.num_heads, logical_chunk, self.head_dim),
                    )
                chunks.append(chunk)
                start += logical_chunk
            attention = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2)
        attention = ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.permute(attention, (1, 0, 2, 3))
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        return ttnn.linear(attention, self.weights["o_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _full_attention_decode(self, hidden_states, page_table, current_positions):
        cache_positions = ttnn.typecast(current_positions, ttnn.int32)
        projected = ttnn.linear(hidden_states, self.weights["packed_qkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        q_and_gate = projected[..., : 2 * q_width]
        q = q_and_gate[..., :q_width]
        gate = q_and_gate[..., q_width:]
        k = projected[..., 2 * q_width : 2 * q_width + kv_width]
        v = projected[..., 2 * q_width + kv_width :]

        fused_qkv = ttnn.concat([q, k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_attention_memory_config,
        )
        q = self._per_head_norm(q, "q_norm")
        k = self._per_head_norm(k, "k_norm")
        q = self._partial_rope_decode(q, current_positions)
        k = self._partial_rope_decode(k, current_positions)

        ttnn.experimental.paged_update_cache(
            self.caches["key"],
            k,
            update_idxs_tensor=cache_positions,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.caches["value"],
            v,
            update_idxs_tensor=cache_positions,
            page_table=page_table,
        )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.caches["key"],
            self.caches["value"],
            cur_pos_tensor=cache_positions,
            page_table_tensor=page_table,
            scale=self.head_dim**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_attention_memory_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        attention = ttnn.linear(attention, self.weights["o_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = attention[:, :, : self.batch, :]
        return ttnn.reshape(attention, (1, 1, self.batch, self.hidden_size))

    def _linear_attention_prefill_chunk(self, hidden_states):
        """Run one 64-token gated-delta chunk with a logarithmic affine scan.

        For each token the recurrent update is ``R' = A R + B``, where
        ``A = d (I - beta k.T k)`` and ``B = beta k.T v``.  Affine transforms
        compose associatively, so a Hillis-Steele scan produces every token
        state in log2(chunk) batched matmuls instead of submitting one decode
        graph per token.
        """
        key_heads = int(self.hf_config.linear_num_key_heads)
        value_heads = int(self.hf_config.linear_num_value_heads)
        key_dim = int(self.hf_config.linear_key_head_dim)
        value_dim = int(self.hf_config.linear_value_head_dim)
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim
        sequence = hidden_states.shape[2]
        groups = self.batch * value_heads

        projected = ttnn.linear(
            hidden_states, self.weights["packed_linear_inputs"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        conv_width = 2 * key_width + value_width
        mixed = projected[..., :conv_width]
        z = projected[..., conv_width : conv_width + value_width]
        beta = projected[..., conv_width + value_width : conv_width + value_width + value_heads]
        decay = projected[..., conv_width + value_width + value_heads :]

        # Stateful depthwise causal convolution, vectorized across the chunk.
        mixed = ttnn.permute(mixed, (0, 1, 3, 2))
        conv_input = ttnn.concat([self.caches["conv"], mixed], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        next_conv_state = conv_input[..., -self.caches["conv"].shape[-1] :]
        # ``conv_state`` stores the last ``kernel`` inputs, while the HF
        # update concatenates it with this chunk and retains the last L
        # *valid* convolution windows.  Their starts are 1..L, not 0..L-1.
        convolved = ttnn.multiply(conv_input[..., 1 : sequence + 1], self.weights["conv"][..., 0:1])
        for kernel_index in range(1, self.caches["conv"].shape[-1]):
            convolved = ttnn.add(
                convolved,
                ttnn.multiply(
                    conv_input[..., kernel_index + 1 : kernel_index + sequence + 1],
                    self.weights["conv"][..., kernel_index : kernel_index + 1],
                ),
            )
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.silu(ttnn.permute(convolved, (0, 1, 3, 2)))

        query = mixed[..., :key_width]
        key = mixed[..., key_width : 2 * key_width]
        value = mixed[..., 2 * key_width :]
        query = ttnn.reshape(query, (self.batch, sequence, key_heads, key_dim))
        key = ttnn.reshape(key, (self.batch, sequence, key_heads, key_dim))
        value = ttnn.reshape(value, (self.batch, sequence, value_heads, value_dim))
        query = ttnn.repeat_interleave(ttnn.permute(query, (0, 2, 1, 3)), value_heads // key_heads, dim=1)
        key = ttnn.repeat_interleave(ttnn.permute(key, (0, 2, 1, 3)), value_heads // key_heads, dim=1)
        value = ttnn.permute(value, (0, 2, 1, 3))
        query = self._l2_norm(query)
        key = self._l2_norm(key)
        query = ttnn.multiply(query, key_dim**-0.5)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(
            self.weights["a"],
            ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])),
        )
        beta = ttnn.permute(
            ttnn.reshape(beta, (self.batch, sequence, value_heads, 1)),
            (0, 2, 1, 3),
        )
        decay = ttnn.exp(
            ttnn.permute(
                ttnn.reshape(decay, (self.batch, sequence, value_heads, 1)),
                (0, 2, 1, 3),
            )
        )
        query = ttnn.reshape(query, (groups, sequence, 1, key_dim))
        key = ttnn.reshape(key, (groups, sequence, 1, key_dim))
        value = ttnn.reshape(value, (groups, sequence, 1, value_dim))
        beta = ttnn.reshape(beta, (groups, sequence, 1, 1))
        decay = ttnn.reshape(decay, (groups, sequence, 1, 1))
        # Projection/bias math above intentionally follows decode's FP32
        # decay policy.  The verified affine scan is BF16, so cast its scalar
        # coefficients explicitly instead of relying on mixed-dtype promotion.
        beta = ttnn.typecast(beta, ttnn.bfloat16)
        decay = ttnn.typecast(decay, ttnn.bfloat16)

        identity = ttnn.repeat(self.weights["linear_identity"], ttnn.Shape([groups, sequence, 1, 1]))
        zero = ttnn.multiply(identity, 0.0)
        key_t = ttnn.transpose(key, -2, -1)
        transform = ttnn.multiply(
            decay,
            ttnn.subtract(
                identity,
                ttnn.multiply(beta, ttnn.matmul(key_t, key)),
            ),
        )
        bias = ttnn.multiply(beta, ttnn.matmul(key_t, value))

        distance = 1
        while distance < sequence:
            previous_transform = ttnn.concat([identity[:, :distance], transform[:, :-distance]], dim=1)
            previous_bias = ttnn.concat([zero[:, :distance], bias[:, :-distance]], dim=1)
            old_transform = transform
            transform = ttnn.matmul(old_transform, previous_transform)
            bias = ttnn.add(ttnn.matmul(old_transform, previous_bias), bias)
            distance *= 2

        initial = ttnn.typecast(self.caches["recurrent"], ttnn.bfloat16)
        initial = ttnn.reshape(initial, (groups, 1, value_dim, value_dim))
        initial = ttnn.repeat(initial, ttnn.Shape([1, sequence, 1, 1]))
        states = ttnn.add(ttnn.matmul(transform, initial), bias)
        final_state = ttnn.reshape(
            states[:, -1:],
            (self.batch, value_heads, value_dim, value_dim),
        )
        ttnn.copy(ttnn.typecast(final_state, ttnn.float32), self.caches["recurrent"])

        output = ttnn.matmul(query, states)
        output = ttnn.reshape(output, (self.batch, value_heads, sequence, value_dim))
        output = ttnn.rms_norm(
            output,
            epsilon=self.eps,
            weight=self.weights["gated_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.permute(
            ttnn.reshape(z, (self.batch, sequence, value_heads, value_dim)),
            (0, 2, 1, 3),
        )
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.reshape(output, (1, self.batch, sequence, value_width))
        return ttnn.linear(output, self.weights["out_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _linear_attention_decode(self, hidden_states):
        key_heads = int(self.hf_config.linear_num_key_heads)
        value_heads = int(self.hf_config.linear_num_value_heads)
        key_dim = int(self.hf_config.linear_key_head_dim)
        value_dim = int(self.hf_config.linear_value_head_dim)
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim

        projected = ttnn.linear(
            hidden_states, self.weights["packed_linear_inputs"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        conv_width = 2 * key_width + value_width
        mixed = projected[..., :conv_width]
        z = projected[..., conv_width : conv_width + value_width]
        beta = projected[..., conv_width + value_width : conv_width + value_width + value_heads]
        decay = projected[..., conv_width + value_width + value_heads :]

        mixed = ttnn.permute(mixed, (0, 2, 3, 1))
        next_conv_state = ttnn.concat(
            [self.caches["conv"][..., 1:], mixed],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mixed = ttnn.sum(
            ttnn.multiply(next_conv_state, self.weights["conv"]),
            dim=-1,
            keepdim=True,
        )
        mixed = ttnn.silu(mixed)
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.permute(mixed, (0, 3, 1, 2))

        query = mixed[..., :key_width]
        key = mixed[..., key_width : 2 * key_width]
        value = mixed[..., 2 * key_width :]
        query = ttnn.reshape(query, (self.batch, 1, key_heads, key_dim))
        key = ttnn.reshape(key, (self.batch, 1, key_heads, key_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.reshape(value, (self.batch, 1, value_heads, value_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))
        repeats = value_heads // key_heads
        query = ttnn.repeat_interleave(query, repeats, dim=1)
        key = ttnn.repeat_interleave(key, repeats, dim=1)
        query = self._l2_norm(query)
        key = self._l2_norm(key)
        query = ttnn.multiply(query, key_dim**-0.5)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(
            self.weights["a"],
            ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])),
        )
        beta = ttnn.reshape(beta, (self.batch, value_heads, 1, 1))
        decay = ttnn.reshape(decay, (self.batch, value_heads, 1, 1))
        decay = ttnn.exp(decay)

        recurrent = ttnn.multiply(self.caches["recurrent"], decay)
        memory_value = ttnn.matmul(key, recurrent)
        delta = ttnn.multiply(ttnn.subtract(value, memory_value), beta)
        update = ttnn.matmul(ttnn.transpose(key, -2, -1), delta)
        recurrent = ttnn.add(recurrent, update)
        output = ttnn.matmul(query, recurrent)
        ttnn.copy(recurrent, self.caches["recurrent"])

        output = ttnn.rms_norm(
            output,
            epsilon=self.eps,
            weight=self.weights["gated_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.reshape(z, (self.batch, value_heads, 1, value_dim))
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.permute(output, (2, 0, 1, 3))
        output = ttnn.reshape(output, (1, 1, self.batch, value_width))
        return ttnn.linear(output, self.weights["out_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
