# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device Phi-3.5 decoder layer.

This stage preserves :class:`FunctionalDecoder`'s public BF16/TILE/DRAM,
paged-cache, LongRoPE, prefill, and trace-safe decode contracts.  It changes
only graph topology; precision, sharding, and matmul program tuning belong to
the later optimized-decoder stage.
"""

from __future__ import annotations

import math

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import (
    PREFILL_SDPA_MAX_SEQ,
    FunctionalDecoder,
)


class FusedDecoder(FunctionalDecoder):
    """Functional-compatible decoder whose MLP executes a fused SwiGLU op."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        full_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(int(device_grid.x) - 1, int(device_grid.y) - 1),
                )
            }
        )
        key_grid = ttnn.num_cores_to_corerangeset(self.batch, device_grid, row_wise=True)
        remaining_grid = full_grid.subtract(key_grid)
        value_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            remaining_grid.ranges()[0].start,
            self.batch,
            remaining_grid,
            True,
        )
        self.decode_key_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(key_grid, [ttnn.TILE_SIZE, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_value_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(value_grid, [ttnn.TILE_SIZE, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _mlp(self, hidden_states):
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        gate_up = ttnn.linear(normalized, self.weights["gate_up"], dtype=ttnn.bfloat16)
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        # Binary-NG applies SiLU to the first input in the multiply kernel,
        # removing the standalone activation dispatch without changing weight
        # precision or folding the activation into the upstream linear.
        activated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.add(hidden_states, ttnn.linear(activated, self.weights["down"], dtype=ttnn.bfloat16))

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table, user_id=0):
        """Run prefill with the fastest measured concat kernel for each batch."""
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[:2] != (1, self.batch) or shape[3] != self.hidden_size:
            raise ValueError(f"prefill hidden_states must be [1,{self.batch},S,{self.hidden_size}], got {shape}")
        seq_len = shape[2]
        if not 1 < seq_len <= self.max_context:
            raise ValueError(f"prefill sequence must be in [2,{self.max_context}], got {seq_len}")
        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = ttnn.linear(normalized, self.weights["qkv"], dtype=ttnn.bfloat16)
        fused = ttnn.reshape(fused, [self.batch, seq_len, 3 * self.hidden_size])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query, key = self._prefill_rope(query, key, seq_len)
        query = ttnn.slice(query, [0, 0, 0, 0], [self.batch, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        value = ttnn.slice(value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        for batch_idx in range(self.batch):
            user_key = ttnn.slice(
                key,
                [batch_idx, 0, 0, 0],
                [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim],
            )
            user_value = ttnn.slice(
                value,
                [batch_idx, 0, 0, 0],
                [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim],
            )
            ttnn.experimental.paged_fill_cache(
                key_cache,
                user_key,
                page_table,
                batch_idx=user_id + batch_idx,
                block_size=self.page_size,
            )
            ttnn.experimental.paged_fill_cache(
                value_cache,
                user_value,
                page_table,
                batch_idx=user_id + batch_idx,
                block_size=self.page_size,
            )
        if seq_len <= PREFILL_SDPA_MAX_SEQ:
            attended = ttnn.transformer.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            attended_chunks = []
            chunk_start = 0
            while chunk_start < seq_len:
                chunk_capacity = PREFILL_SDPA_MAX_SEQ if chunk_start == 0 else 4 * ttnn.TILE_SIZE
                chunk_len = min(chunk_capacity, seq_len - chunk_start)
                padded_len = math.ceil(chunk_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                query_chunk = ttnn.slice(
                    query,
                    [0, 0, chunk_start, 0],
                    [self.batch, self.num_heads, chunk_start + chunk_len, self.head_dim],
                )
                if padded_len != chunk_len:
                    query_chunk = ttnn.pad(
                        query_chunk,
                        [(0, 0), (0, 0), (0, padded_len - chunk_len), (0, 0)],
                        value=0.0,
                    )
                if chunk_start == 0 and chunk_len == PREFILL_SDPA_MAX_SEQ:
                    prefix_key = ttnn.slice(
                        key,
                        [0, 0, 0, 0],
                        [self.batch, self.num_kv_heads, chunk_len, self.head_dim],
                    )
                    prefix_value = ttnn.slice(
                        value,
                        [0, 0, 0, 0],
                        [self.batch, self.num_kv_heads, chunk_len, self.head_dim],
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        prefix_key,
                        prefix_value,
                        is_causal=True,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    mask = self._offset_causal_mask(
                        chunk_start=chunk_start,
                        query_len=padded_len,
                        key_len=seq_len,
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        key,
                        value,
                        attn_mask=mask,
                        is_causal=False,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                if padded_len != chunk_len:
                    output_chunk = ttnn.slice(
                        output_chunk,
                        [0, 0, 0, 0],
                        [self.batch, self.num_heads, chunk_len, self.head_dim],
                    )
                attended_chunks.append(output_chunk)
                chunk_start += chunk_len
            attended = attended_chunks[0] if len(attended_chunks) == 1 else ttnn.concat(attended_chunks, dim=2)
        # The dedicated kernel wins at serving batch 32, but its fixed launch
        # cost exceeds the saving at batch 1.
        attended = (
            ttnn.experimental.nlp_concat_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.batch > 1
            else ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        projected = ttnn.linear(attended, self.weights["o_proj"], dtype=ttnn.bfloat16)
        projected = ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size])
        return self._mlp(ttnn.add(residual, projected))

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        use_long_rope,
    ):
        """Run decode with the paired K/V cache writes fused into one op."""
        shape = tuple(hidden_states.shape)
        if shape != (1, 1, self.batch, self.hidden_size):
            raise ValueError(f"decode hidden_states must be [1,1,{self.batch},{self.hidden_size}], got {shape}")
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape [{self.batch}], got {tuple(current_positions.shape)}")
        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = ttnn.linear(normalized, self.weights["qkv"], dtype=ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query, key = self._decode_rope(query, key, current_positions, use_long_rope=use_long_rope)
        if self.batch == 1:
            # At batch 1 the paired update saves a full cache-write dispatch
            # and the one extra V reshard is materially cheaper. At serving
            # batch 32 the reshard costs more than the fused write saves on
            # Blackhole, so retain the faster pair of dedicated paged writes.
            key = ttnn.to_memory_config(key, self.decode_key_memory_config)
            value = ttnn.to_memory_config(value, self.decode_value_memory_config)
            ttnn.experimental.paged_fused_update_cache(
                key_cache,
                key,
                value_cache,
                value,
                update_idxs_tensor=current_positions,
                page_table=page_table,
            )
        else:
            ttnn.experimental.paged_update_cache(
                key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
            )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            cur_pos_tensor=current_positions,
            page_table_tensor=page_table,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
        if self.batch < ttnn.TILE_SIZE:
            attended = ttnn.slice(attended, [0, 0, 0, 0], [1, 1, self.batch, self.hidden_size])
        projected = ttnn.linear(attended, self.weights["o_proj"], dtype=ttnn.bfloat16)
        projected = ttnn.reshape(projected, [1, 1, self.batch, self.hidden_size])
        return self._mlp(ttnn.add(residual, projected))
