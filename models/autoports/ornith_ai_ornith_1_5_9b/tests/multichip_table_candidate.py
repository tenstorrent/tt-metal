# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Setup-owned request offsets and batched page-table padding/addition."""

import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


class BatchedTableCandidate(MultichipDecoder):
    def allocate_batch_state(self, *, batch, context=None):
        states = super().allocate_batch_state(batch=batch, context=context)
        if self.kind == "full_attention" and batch > 1:
            if not hasattr(self, "pool_offset_tensors"):
                self.pool_offset_tensors = {}
            self.pool_offset_tensors[id(states[0].key)] = ttnn.from_torch(
                torch.tensor([[st.page_offset] for st in states], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        return states

    def full_attention_decode_batch(self, packed, states, current_positions, page_tables):
        """Batch SDPA with private cache-pool packing and explicit scatter-back."""
        n = len(states)
        if not (
            id(states[0].key) in getattr(self, "pool_offset_tensors", {})
            and all(st.key is states[0].key and st.page_offset == i * st.page_count for i, st in enumerate(states))
            and n * states[0].page_count == states[0].key.shape[0]
            and len({table.shape[-1] for table in page_tables}) == 1
        ):
            return super().full_attention_decode_batch(packed, states, current_positions, page_tables)
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
                ranges.add(
                    ttnn.CoreRange(ttnn.CoreCoord(0, y_offset + rows), ttnn.CoreCoord(tail - 1, y_offset + rows))
                )
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
        key, value = states[0].key, states[0].value
        pools = [(key, value)]
        offsets = self.pool_offset_tensors[id(key)]
        local_table = ttnn.concat(page_tables, dim=0)
        attention_table = local_table
        width = local_table.shape[-1]
        if width % 8:
            attention_table = ttnn.pad(local_table, [(0, 0), (0, 8 - width % 8)], 0)
        table = ttnn.add(local_table, offsets)
        attention_table = table if attention_table is local_table else ttnn.add(attention_table, offsets)
        ttnn.experimental.paged_fused_update_cache(key, k, value, v, update_idxs_tensor=positions, page_table=table)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            key,
            value,
            attention_table,
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
        if len(pools) > 1:
            offset = 0
            for old_key, old_value in pools:
                end = offset + old_key.shape[0]
                ttnn.slice(key, (offset, 0, 0, 0), (end, 1, 32, 256), (1, 1, 1, 1), output_tensor=old_key)
                ttnn.slice(value, (offset, 0, 0, 0), (end, 1, 32, 256), (1, 1, 1, 1), output_tensor=old_value)
                offset = end
        out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return out
