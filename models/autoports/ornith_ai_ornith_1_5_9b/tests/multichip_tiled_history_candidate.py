# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent tiled convolution taps in [tap,1,request,channel] order."""
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fusion_reference import MultichipDecoder, DecoderState

class TiledHistoryCandidate(MultichipDecoder):
    fused_conv = True
    fused_pad = True
    gdn_depth = None
    def allocate_batch_state(self, *, batch, context=None):
        states = super().allocate_batch_state(batch=batch, context=context)
        if batch > 1 and self.kind == "linear_attention":
            conv = ttnn.zeros((3, 1, batch, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for state in states:
                state.conv = conv
        return states

    def read_linear_state(self, state, name):
        row = state.state_row
        value = getattr(state, name)
        if row is None:
            return value
        if name == "conv":
            return ttnn.to_layout(ttnn.permute(value[:, :, row:row+1, :], (2, 1, 0, 3)), ttnn.ROW_MAJOR_LAYOUT)
        return value[row:row+1]

    def write_linear_state(self, state, name, value):
        if state.state_row is not None and name == "conv":
            # The cache updater stages a full padded user tile. Width2048
            # needs5.36MiB/core; split setup-time writes to256-channel blocks.
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
            pieces = []
            for start in range(0, 2048, 256):
                block = state.conv[:, :, :, start:start+256]
                ttnn.update_cache(block, value[:, :, :, start:start+256], state.state_row)
                pieces.append(block)
            ttnn.copy(ttnn.concat(pieces, dim=3), state.conv)
        else:
            super().write_linear_state(state, name, value)

    def linear_attention(self, x, state, *, decode, packed=None, project=True):
        if state.state_row is None:
            return super().linear_attention(x, state, decode=decode, packed=packed, project=project)
        local = DecoderState(conv=self.read_linear_state(state, "conv"), recurrent=self.read_linear_state(state, "recurrent"))
        out = self.linear_attention(x, local, decode=decode, packed=packed, project=project)
        self.write_linear_state(state, "conv", local.conv)
        self.write_linear_state(state, "recurrent", local.recurrent)
        return out

    def pooled_linear_attention_decode_batch(self, packed, states):
        """Neutral tail tokens and disjoint head waves update pooled state in place."""
        n = len(states)
        raw = packed[:, :, :, :2048]
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8), **({"pad_value": 0.0} if self.fused_pad else {}))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.reshape(
            ttnn.multiply(
                ttnn.add(a, self.dt_bias),
                self.decay,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            ),
            (n, 1, 8),
            **({"pad_value": 0.0} if self.fused_pad else {}),
        )
        history = states[0].conv
        conv = ttnn.multiply(ttnn.typecast(raw, ttnn.float32), self.float_taps[3])
        for j in range(3):
            old = history[j : j + 1, :, :, :]
            conv = (ttnn.addcmul(conv, ttnn.typecast(old, ttnn.float32), self.float_taps[j])
                    if self.fused_conv else ttnn.add(conv, ttnn.multiply(ttnn.typecast(old, ttnn.float32), self.float_taps[j])))
        conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
        new_history = ttnn.concat([history[1:, :, :, :], raw], dim=0)
        ttnn.copy(new_history, states[0].conv)

        def padded_flat(value, width):
            value = ttnn.reshape(value, (n, 1, width), **({"pad_value": 0.0} if self.fused_pad else {}))
            return (ttnn.reshape(value, ttnn.Shape([n, 32, width]), value.padded_shape) if self.fused_pad
                    else ttnn.pad(value, [(0, 0), (0, 31), (0, 0)], 0.0))

        q = padded_flat(conv[:, :, :, :512], 512)
        k = padded_flat(conv[:, :, :, 512:1024], 512)
        v = padded_flat(conv[:, :, :, 1024:], 1024)
        g = (ttnn.reshape(g, ttnn.Shape([n, 32, 8]), g.padded_shape) if self.fused_pad
             else ttnn.pad(g, [(0, 0), (0, 31), (0, 0)], 0.0))
        beta = (ttnn.reshape(beta, ttnn.Shape([n, 32, 8]), beta.padded_shape) if self.fused_pad
             else ttnn.pad(beta, [(0, 0), (0, 31), (0, 0)], 0.0))
        recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
        scratch = ttnn.empty(
            (n * 8, 1, 32, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wave_heads = self.decode_wave_heads
        for head_offset in range(0, n * 8, wave_heads):
            out, _ = ttnn._ttnn.operations.transformer.chunk_gdn_fused(
                q,
                k,
                v,
                g,
                beta,
                self.gdn_constants["eye"],
                self.gdn_constants["tril"],
                self.gdn_constants["ones"],
                self.gdn_constants["masks"],
                recurrent,
                chunk_size=32,
                output_final_state=True,
                compute_kernel_config=self.compute,
                program_config=ttnn.ChunkGdnFusedProgramConfig(
                    num_receivers=1,
                    handoff_depth=self.gdn_depth,
                    **({"num_producers": self.decode_wave_producers} if self.decode_wave_producers else {}),
                ),
                num_heads=min(wave_heads, n * 8 - head_offset),
                head_offset=head_offset,
                v_flat=True,
                HV=8,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=4,
                o_out=scratch,
                final_state_out=recurrent,
            )
        out = ttnn.permute(ttnn.reshape(out, (n, 8, 32, 128))[:, :, :1, :], (0, 2, 1, 3))
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out

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
            and states[0].conv.shape[2] == n
        )
        independent_linear = all(getattr(st, "state_row", None) is None for st in states)
        if n > 1 and (self.kind == "full_attention" or pooled_linear or independent_linear):
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

def main():
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_tiled_history_stack as stack
    stack.MultichipDecoder = TiledHistoryCandidate
    stack.main()

if __name__ == "__main__":
    main()
