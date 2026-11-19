# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched full attention with explicit private-cache pool packing and scatter-back."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_recurrent_candidate import RecurrentCandidate


class AttentionBatchCandidate(RecurrentCandidate):
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
            key = ttnn.concat([st.key for st in states], dim=0)
            value = ttnn.concat([st.value for st in states], dim=0)
            offset = 0
            tables = []
            for st, table in zip(states, page_tables):
                tables.append(ttnn.add(table, offset) if offset else table)
                offset += st.key.shape[0]
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
            offset = 0
            for st in states:
                end = offset + st.key.shape[0]
                ttnn.slice(key, (offset, 0, 0, 0), (end, 1, 32, 256), (1, 1, 1, 1), output_tensor=st.key)
                ttnn.slice(value, (offset, 0, 0, 0), (end, 1, 32, 256), (1, 1, 1, 1), output_tensor=st.value)
                offset = end
            out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
            return self.finish(x, self.reduce(self.linear(out, "self_attn.o_proj")))
        finally:
            self._grouped = False
