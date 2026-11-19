# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-stack partial Q/K norm/RoPE fusion candidate."""
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_sdpa_reference import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_partial_norm_rope import partial_norm_rope

class PartialRopeCandidate(MultichipDecoder):
    def full_attention_decode_batch(self, packed, states, current_positions, page_tables):
        """Batch SDPA with private cache-pool packing and explicit scatter-back."""
        n = len(states)
        head_memory = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(x=8, y=4),
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        q8, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            packed,
            num_heads=8,
            num_kv_heads=1,
            overlap_qk_coregrid=True,
            memory_config=head_memory,
        )
        # Packed Q/G occupies eight query-sized sections. Only the first four
        # are logical Q heads; the view retains each user's padded32-head stride.
        q = ttnn.reshape(q8, ttnn.Shape([1, n, 4, 256]), ttnn.Shape([1, n, 32, 256]))
        gate = packed[:, :, :, 1024:2048]
        positions = ttnn.concat(current_positions, dim=0)
        rp = ttnn.typecast(positions, ttnn.uint32)
        if n < 32:
            rp = ttnn.pad(rp, [(0, 32 - n)], 0)
        values = ttnn.reshape(ttnn.embedding(rp, self.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, 32, 128))[
            :, :, :n, :
        ]
        cos, sin = values[:, :, :, :64], values[:, :, :, 64:]
        q, k = partial_norm_rope(q, k, self.weights["self_attn.q_norm.weight"],
                                  self.weights["self_attn.k_norm.weight"], cos, sin, self.config.rms_norm_eps)


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
        ordered_pool = (
            states[0].page_offsets is not None
            and all(
                st.key is states[0].key and st.value is states[0].value and st.page_offset == i * st.page_count
                for i, st in enumerate(states)
            )
            and n * states[0].page_count == states[0].key.shape[0]
            and len({table.shape[-1] for table in page_tables}) == 1
        )
        if ordered_pool:
            key, value = states[0].key, states[0].value
            pools = [(key, value)]
            local_table = ttnn.concat(page_tables, dim=0)
            attention_table = local_table
            width = local_table.shape[-1]
            if width % 4:
                attention_table = ttnn.pad(local_table, [(0, 0), (0, 4 - width % 4)], 0)
            table = ttnn.add(local_table, states[0].page_offsets)
            attention_table = (
                table if attention_table is local_table else ttnn.add(attention_table, states[0].page_offsets)
            )
        else:
            pools, pool_offsets, total_pages = [], {}, 0
            for state in states:
                identity = (id(state.key), id(state.value))
                if identity not in pool_offsets:
                    pools.append((state.key, state.value))
                    pool_offsets[identity] = total_pages
                    total_pages += state.key.shape[0]
            key = pools[0][0] if len(pools) == 1 else ttnn.concat([pair[0] for pair in pools], dim=0)
            value = pools[0][1] if len(pools) == 1 else ttnn.concat([pair[1] for pair in pools], dim=0)
            tables = []
            max_pages = max(table.shape[-1] for table in page_tables)
            for state, table in zip(states, page_tables):
                # Different allocated contexts are valid. Padded local page zero is
                # inactive for valid positions within each request's own context.
                if table.shape[-1] < max_pages:
                    table = ttnn.pad(table, [(0, 0), (0, max_pages - table.shape[-1])], 0)
                offset = pool_offsets[(id(state.key), id(state.value))] + getattr(state, "page_offset", 0)
                tables.append(ttnn.add(table, offset) if offset else table)
            table = ttnn.concat(tables, dim=0)
            attention_table = table
            if max_pages % 4:
                attention_pages = ((max_pages + 3) // 4) * 4
                attention_rows = []
                for state, local_table in zip(states, page_tables):
                    padded = ttnn.pad(local_table, [(0, 0), (0, attention_pages - local_table.shape[-1])], 0)
                    offset = pool_offsets[(id(state.key), id(state.value))] + state.page_offset
                    attention_rows.append(ttnn.add(padded, offset) if offset else padded)
                attention_table = ttnn.concat(attention_rows, dim=0)
        ttnn.experimental.paged_fused_update_cache(key, k, value, v, update_idxs_tensor=positions, page_table=table)
        extra = {}
        if n > 1:
            assert 2 <= n <= 32 and self.device.compute_with_storage_grid_size().x >= 11
            w, h = 11, 10
            assert h >= (n + 7) // 8
            # The first n compute cores must match the existing Q shard order.
            # List order is intentional: do not merge or sort these ranges.
            ranges = []
            rows, tail = divmod(n, 8)

            def add(x0, y0, x1, y1):
                if x0 <= x1 and y0 <= y1:
                    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)))

            add(0, 0, 7, rows - 1)
            if tail:
                add(0, rows, tail - 1, rows)
            add(8, 0, w - 1, rows - 1)
            if tail:
                add(tail, rows, w - 1, rows)
            add(0, rows + bool(tail), w - 1, h - 1)
            extra["sub_core_grids"] = ttnn.CoreRangeSet(ranges)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            key,
            value,
            attention_table,
            cur_pos_tensor=positions,
            scale=1 / 16,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(11, 10), q_chunk_size=32, k_chunk_size=128, **extra
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

def main():
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder=PartialRopeCandidate
    stack.main()

if __name__=="__main__":main()
