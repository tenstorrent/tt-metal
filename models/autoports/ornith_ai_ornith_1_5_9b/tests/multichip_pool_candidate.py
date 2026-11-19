# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Setup-owned pooled KV with disjoint per-request pages; no per-step pool copies."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_stage4_reference import DecoderState
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_attention_batch_candidate import AttentionBatchCandidate


class PoolCandidate(AttentionBatchCandidate):
    def allocate_batch_state(self, *, batch, context):
        assert 1 <= batch <= 32 and 0 < context <= self.config.max_position_embeddings and context % 32 == 0
        if self.kind == "linear_attention":
            return [self.allocate_state(context=context) for _ in range(batch)]
        pages = context // 32
        def empty():
            return ttnn.zeros((batch * pages, 1, 32, 256), dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT,
                              device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key, value = empty(), empty()
        states = [DecoderState(key=key, value=value) for _ in range(batch)]
        for i, state in enumerate(states):
            state.page_offset, state.page_count = i * pages, pages
        return states

    def full_attention(self, x, state, *, page_table, **kwargs):
        if getattr(state, "page_offset", 0):
            page_table = ttnn.add(page_table, state.page_offset)
        return super().full_attention(x, state, page_table=page_table, **kwargs)

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        n = x.shape[2]
        if self.kind != "full_attention" or n == 1:
            return super().decode_batch_packed(
                x, states=states, current_positions=current_positions, page_tables=page_tables
            )
        self._grouped = True
        try:
            x = ttnn.to_memory_config(x, self.ar_mem)
            packed = self.linear(self.norm(x, "input_layernorm"), "packed_input")
            q = self.norm(ttnn.reshape(packed[:, :, :, :1024], (1, n, 4, 256)), "self_attn.q_norm")
            gate = packed[:, :, :, 1024:2048]
            k = self.norm(ttnn.reshape(packed[:, :, :, 2048:2304], (1, n, 1, 256)), "self_attn.k_norm")
            v = ttnn.reshape(packed[:, :, :, 2304:2560], (1, n, 1, 256))
            positions = ttnn.concat(current_positions, dim=0)
            rp = ttnn.typecast(positions, ttnn.uint32)
            if n < 32:
                rp = ttnn.pad(rp, [(0, 32 - n)], 0)
            values = ttnn.reshape(ttnn.embedding(rp, self.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, 32, 128))[
                :, :, :n, :
            ]
            cos, sin = values[:, :, :, :64], values[:, :, :, 64:]
            q = ttnn.permute(self.rope(ttnn.permute(q, (0, 2, 1, 3)), cos, sin), (0, 2, 1, 3))
            k = ttnn.permute(self.rope(ttnn.permute(k, (0, 2, 1, 3)), cos, sin), (0, 2, 1, 3))

            def shard(y_offset):
                ranges = set()
                rows, tail = divmod(n, 8)
                if rows:
                    ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, y_offset), ttnn.CoreCoord(7, y_offset + rows - 1)))
                if tail:
                    ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, y_offset + rows), ttnn.CoreCoord(tail - 1, y_offset + rows)))
                # SDPA enumerates ranges before row-major cores within each range.
                # Singleton ranges sort by x first and permute independent users.
                cores = ttnn.CoreRangeSet(ranges)
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, (32, 256), ttnn.ShardOrientation.ROW_MAJOR),
                )

            q = ttnn.to_memory_config(q, shard(0))
            k = ttnn.to_memory_config(k, shard(0))
            v = ttnn.to_memory_config(v, shard(4))
            assert all(st.key is states[0].key and st.value is states[0].value for st in states)
            key, value = states[0].key, states[0].value
            tables = [ttnn.add(table, st.page_offset) if st.page_offset else table
                      for st, table in zip(states, page_tables)]
            table = ttnn.concat(tables, dim=0)
            ttnn.experimental.paged_fused_update_cache(key, k, value, v, update_idxs_tensor=positions, page_table=table)
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                key,
                value,
                table,
                cur_pos_tensor=positions,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(out, shard(0)),
                num_heads=4,
                **({"sub_core_grids": shard(0).shard_spec.grid} if n >= 4 else {}),
            )
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)[:, :, :n, :]
            out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
            return self.finish(x, self.reduce(self.linear(out, "self_attn.o_proj")))
        finally:
            self._grouped = False
