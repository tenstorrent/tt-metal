# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Use the tested tiled pool and fused recurrence at logical batch one."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder, DecoderState

class PooledOne(MultichipDecoder):
    def allocate_batch_state(self, *, batch, context=None):
        if batch != 1 or self.kind != "linear_attention":
            return super().allocate_batch_state(batch=batch,context=context)
        context = context or self.config.max_position_embeddings
        assert 0 < context <= self.config.max_position_embeddings and context % 32 == 0
        conv = ttnn.zeros((3,1,1,2048),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=self.device,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        recurrent = ttnn.zeros((1,8,128,128),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=self.device,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [DecoderState(conv=conv,recurrent=recurrent,state_row=0)]

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        """TP4 BF16 [1,1,B,4096], 1<=B<=32, local L1 residual across layers.

        Request attention/cache state stays independent. Dense projections, norms,
        residual adds and reductions share the physical tile. The caller packs at
        stack entry and unpacks only at its exit, outside decoder layer boundaries.
        """
        n = x.shape[2]
        assert self.native_local and 1 <= n <= 32
        assert n == len(states) == len(current_positions) == len(page_tables)
        regions = [
            (
                (id(st.conv), id(st.recurrent), st.state_row)
                if self.kind == "linear_attention"
                else (id(st.key), id(st.value), st.page_offset)
            )
            for st in states
        ]
        assert len(set(regions)) == n, "A decode batch must contain distinct request state regions"
        x = ttnn.to_memory_config(x, self.ar_mem)
        packed = self.decode_linear(self.decode_norm(x, "input_layernorm"), "packed_input")
        pooled_linear = (
            self.kind == "linear_attention"
            and all(
                st.conv is states[0].conv
                and st.recurrent is states[0].recurrent
                and getattr(st, "state_row", None) == i
                for i, st in enumerate(states)
            )
            and tuple(states[0].conv.shape) == (3, 1, n, 2048)
            and states[0].conv.layout == ttnn.TILE_LAYOUT
            and states[0].recurrent.shape[0] == n
        )
        independent_linear = all(getattr(st, "state_row", None) is None for st in states)
        if pooled_linear or (n > 1 and (self.kind == "full_attention" or independent_linear)):
            if self.kind == "full_attention":
                local = self.full_attention_decode_batch(packed, states, current_positions, page_tables)
            elif pooled_linear:
                local = self.pooled_linear_attention_decode_batch(packed, states)
            else:
                local = self.linear_attention_decode_batch(packed, states)
        else:
            local = []
            for i, (state, position, table) in enumerate(zip(states, current_positions, page_tables)):
                row = packed[:, :, i : i + 1, :]
                local.append(
                    self.linear_attention(row, state, decode=True, packed=row, project=False)
                    if self.kind == "linear_attention"
                    else self.full_attention(
                        row, state, page_table=table, position=position, decode=True, packed=row, project=False
                    )
                )
            local = ttnn.concat(local, dim=2) if n > 1 else local[0]
        name = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
        return self.decode_finish(x, self.decode_reduce(self.decode_linear(local, name)))


if __name__ == "__main__":
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder=PooledOne
    stack.main()
