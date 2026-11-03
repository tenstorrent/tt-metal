# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused RS plus distributed grouped norm, with final decode precision/layout."""
import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_distributed_norm import (
    MultichipDecoder as Distributed,
)
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_stage5_fused_rs import MultichipDecoder as Fused
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_stage5_multicore_combined import (
    MultichipDecoder as Multicore,
)


class MultichipDecoder(Fused, Distributed, Multicore):
    def _row_project(self, x, name):
        if not self.options.get("fused_row", True):
            return self._reduce(self._linear(x, name))
        return super()._row_project(x, name)

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
        )
        x = ttnn.add(residual, self._row_project(gated, "self_attn.o_proj"))
        normed = self._norm(x, "post_attention_layernorm")
        return ttnn.add(x, self._mlp(normed, valid_tokens=valid_tokens))

    def decode_forward(self, hidden_states, *, cos, sin, page_table, current_pos, kv_cache, _prefill=False):
        """Device-only traceable decode; current_pos and page_table are mutable inputs.

        cos/sin: [1,B,1,128]. No host reads, conversions, cache construction or
        state reset occurs in this call. The output is [1,1,B,2560].
        """
        b = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, b, 640 if self.residual_sharded else 2560) or not 1 <= b <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1<=B<=32")
        padded = math.ceil(b / 32) * 32
        x = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        normed = self._norm(x, "input_layernorm")
        packed = self._linear(
            normed,
            (
                "attention_decode"
                if not _prefill and b == 1 and "attention_decode.weight" in self.decode_weights
                else "attention_packed"
            ),
        )
        qkv_width = 1280 if self.sparse else 1536
        gate_logits = packed[..., qkv_width : qkv_width + 1024]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat(
                [qkv, self._experts(normed, value=True, valid_tokens=b, logits=packed[..., qkv_width + 1024 :])], dim=-1
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
            qkv[:, :, :b, :], num_heads=8, num_kv_heads=2, memory_config=mem
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
        if page_table.shape[1] * kv_cache[0].shape[2] >= 262144:
            # The long page-table CB and k_chunk1024 nearly fill L1. Release
            # the transient query shard before SDPA; capacity stays unchanged.
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        attention_compute, attention_program = self._decode_attention_config(b, page_table, kv_cache[0])
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            cur_pos_tensor=current_pos,
            compute_kernel_config=attention_compute,
            program_config=attention_program,
        )
        concat_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Nonrectangular user grids select the subcore-grid factory, which
        # requires an explicit allowed output grid. Keep exactly B input shards.
        concat_options = {}
        if b > 8 and b % 8:
            concat_options["sub_core_grids"] = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}
            )
        out = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(out, concat_mem), num_heads=8, **concat_options
        )
        return self._finish(x, gate_logits, out, valid_tokens=b)[:, :, :b, :]
