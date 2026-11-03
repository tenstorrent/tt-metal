# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-page-table attention query placement comparison."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def decode_forward(self, hidden_states, *, cos, sin, page_table, current_pos, kv_cache):
        """Device-only traceable decode; current_pos and page_table are mutable inputs.

        cos/sin: [1,B,1,128]. No host reads, conversions, cache construction or
        state reset occurs in this call. The output is [1,1,B,2560].
        """
        b = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, b, 2560) or not 1 <= b <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1<=B<=32")
        padded = math.ceil(b / 32) * 32
        x = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        normed = self._norm(x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 5120 if self.sparse else 6144
        gate_logits = packed[..., qkv_width : qkv_width + 4096]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat(
                [qkv, self._experts(normed, value=True, valid_tokens=b, logits=packed[..., qkv_width + 4096 :])], dim=-1
            )
        # SDPA's sharded reader uses its 8-wide reducer grid, not the physical
        # device width. Producer and consumer must agree on each user's core.
        grid = ttnn.num_cores_to_corerangeset(b, ttnn.CoreCoord(8, 8), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv[:, :, :b, :], num_heads=32, num_kv_heads=8, memory_config=mem
        )
        if b > 8 and b % 8:
            # Native rotary launches the grid's bounding rectangle. Back every
            # launched core, including the inactive final-row slots.
            rectangle = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, (b - 1) // 8))})
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(rectangle, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            q, k = ttnn.to_memory_config(q, mem), ttnn.to_memory_config(k, mem)
        cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        q, k = self._rope(q, cos, sin, decode=True), self._rope(k, cos, sin, decode=True)
        self._update_cache(k, v, current_pos, page_table, kv_cache)
        del k, v, cos, sin
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=tuple(self.options.get("sdpa_grid", [8, 8])),
                q_chunk_size=0,
                k_chunk_size=self.options.get("sdpa_k", 1024),
                exp_approx_mode=self.options.get("sdpa_approx", False),
            ),
        )
        out = ttnn.reshape(out, (1, 1, b, 4096))
        out = ttnn.pad(out, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        return self._finish(x, gate_logits, out, valid_tokens=b)[:, :, :b, :]
