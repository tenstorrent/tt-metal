# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cache dtype and SDPA geometry candidates; fill dtype matches destination."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_prefill import PrefillCandidate


class CacheCandidate(PrefillCandidate):
    decode_heads = False
    fused_cache = False
    tiled_lookup = False
    kv_dtype = ttnn.bfloat16
    sdpa_q_chunk = 32
    sdpa_k_chunk = 32

    def allocate_state(self, **kwargs):
        state = super().allocate_state(**kwargs)
        if self.kind == "full_attention" and self.kv_dtype != ttnn.bfloat16:
            state.key = ttnn.typecast(state.key, self.kv_dtype)
            state.value = ttnn.typecast(state.value, self.kv_dtype)
        return state

    def full_attention(self, x, state, *, page_table, position, decode):
        n = x.shape[2]
        packed = self.linear(x, "packed_input")
        if decode and self.decode_heads:
            qg, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                packed, num_heads=32, num_kv_heads=4, memory_config=self.shard
            )
            q = self.norm(ttnn.to_memory_config(qg, ttnn.DRAM_MEMORY_CONFIG)[:, :, :16, :], "self_attn.q_norm")
            k = self.norm(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), "self_attn.k_norm")
            gate = packed[:, :, :, 4096:8192]
        elif decode:
            q = self.norm(ttnn.reshape(packed[:, :, :, :4096], (1, n, 16, 256)), "self_attn.q_norm")
            gate = packed[:, :, :, 4096:8192]
            k = self.norm(ttnn.reshape(packed[:, :, :, 8192:9216], (1, n, 4, 256)), "self_attn.k_norm")
            v = ttnn.reshape(packed[:, :, :, 9216:10240], (1, n, 4, 256))
        else:
            qg, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn.reshape(packed, (1, n, 10240)),
                num_heads=32,
                num_kv_heads=4,
                transpose_key=False,
            )
            q = self.norm(qg[:, :16, :, :], "self_attn.q_norm")
            k = self.norm(k, "self_attn.k_norm")
            gate = ttnn.reshape(ttnn.permute(qg[:, 16:32, :, :], (0, 2, 1, 3)), (1, 1, n, 4096))
        if decode:
            rope_position = ttnn.typecast(position, ttnn.uint32)
            if self.tiled_lookup:
                rope_position = ttnn.pad(rope_position, [(0, 31)], 0)
            rope_values = ttnn.reshape(
                ttnn.embedding(rope_position, self.rope_decode, layout=ttnn.TILE_LAYOUT),
                (1, 1, 32 if self.tiled_lookup else 1, 128),
            )
            rope_values = rope_values[:, :, :1, :]
            cos, sin = rope_values[:, :, :, :64], rope_values[:, :, :, 64:]
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            q = ttnn.to_memory_config(q, self.shard)
            k = ttnn.to_memory_config(k, self.shard)
            value_shard = self.shard
            if self.fused_cache:
                value_shard = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                        (32, 256),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
            v = ttnn.to_memory_config(v, value_shard)
            if self.fused_cache:
                ttnn.experimental.paged_fused_update_cache(
                    state.key, k, state.value, v, update_idxs_tensor=position, page_table=page_table
                )
            else:
                ttnn.experimental.paged_update_cache(state.key, k, update_idxs_tensor=position, page_table=page_table)
                ttnn.experimental.paged_update_cache(state.value, v, update_idxs_tensor=position, page_table=page_table)
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                state.key,
                state.value,
                page_table,
                cur_pos_tensor=position,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=self.sdpa_q_chunk,
                    k_chunk_size=self.sdpa_k_chunk,
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(out, self.shard), num_heads=16)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)[:, :, :1, :]
        else:
            cos = ttnn.reshape(self.cos[position : position + n, :], (1, 1, n, 64))
            sin = ttnn.reshape(self.sin[position : position + n, :], (1, 1, n, 64))
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            local_table = page_table[:, position // 32 : (position + n) // 32]
            ttnn.experimental.paged_fill_cache(state.key, ttnn.typecast(k, self.kv_dtype), local_table)
            ttnn.experimental.paged_fill_cache(state.value, ttnn.typecast(v, self.kv_dtype), local_table)
            attention_table = page_table
            if page_table.shape[1] % 8:
                attention_table = ttnn.pad(page_table, [(0, 0), (0, 8 - page_table.shape[1] % 8)], 0)
            out = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                state.key,
                state.value,
                attention_table,
                chunk_start_idx_tensor=self.positions[position : position + 1],
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=self.sdpa_q_chunk,
                    k_chunk_size=self.sdpa_k_chunk,
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, n, 4096))
        return self.linear(
            ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), "self_attn.o_proj"
        )
