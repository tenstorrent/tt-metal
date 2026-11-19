# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch chunk-GDN: native independent B axis, internally neutral padded tokens."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_full_pool_candidate import FullPoolCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batched_candidate import BatchedCandidate


class ChunkPoolCandidate(FullPoolCandidate):
    wave_heads = 32

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        if self.kind != "linear_attention":
            return super().decode_batch_packed(x, states=states, current_positions=current_positions, page_tables=page_tables)
        if x.shape[2] == 1:
            return BatchedCandidate.decode_batch_packed(self, x, states=states, current_positions=current_positions, page_tables=page_tables)
        if not hasattr(states[0], "state_row"):
            return super().decode_batch_packed(x, states=states, current_positions=current_positions, page_tables=page_tables)
        assert all(st.conv is states[0].conv and st.recurrent is states[0].recurrent and st.state_row == i
                   for i, st in enumerate(states)) and len(states) == states[0].conv.shape[0]
        n = x.shape[2]
        assert 1 <= n <= 32 and n == len(states)
        self._grouped = True
        try:
            x = ttnn.to_memory_config(x, self.ar_mem)
            packed = self.linear(self.norm(x, "input_layernorm"), "packed_input")
            raw = packed[:, :, :, :2048]
            z = packed[:, :, :, 2048:3072]
            beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8))
            a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
            g = ttnn.reshape(
                ttnn.multiply(
                    ttnn.add(a, self.dt_bias),
                    self.decay,
                    input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
                ),
                (n, 1, 8),
            )
            history = states[0].conv
            conv = ttnn.multiply(ttnn.typecast(raw, ttnn.float32), self.float_taps[3])
            for j in range(3):
                old = ttnn.to_layout(ttnn.reshape(history[:, :, j : j + 1, :], (1, 1, n, 2048)), ttnn.TILE_LAYOUT)
                conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(old, ttnn.float32), self.float_taps[j]))
            conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
            new_history = ttnn.concat(
                [history[:, :, 1:, :], ttnn.reshape(ttnn.to_layout(raw, ttnn.ROW_MAJOR_LAYOUT), (n, 1, 1, 2048))], dim=2
            )
            ttnn.copy(new_history, states[0].conv)
            def padded_flat(value, width):
                value = ttnn.reshape(value, (n, 1, width))
                return ttnn.pad(value, [(0, 0), (0, 31), (0, 0)], 0.0)

            q = padded_flat(conv[:, :, :, :512], 512)
            k = padded_flat(conv[:, :, :, 512:1024], 512)
            v = padded_flat(conv[:, :, :, 1024:], 1024)
            g = ttnn.pad(g, [(0, 0), (0, 31), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 31), (0, 0)], 0.0)
            recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
            scratch = ttnn.empty((n * 8, 1, 32, 128), dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            wave_heads = self.wave_heads or 24
            for head_offset in range(0, n * 8, wave_heads):
                out, _ = ttnn._ttnn.operations.transformer.chunk_gdn_fused(
                    q, k, v, g, beta, self.gdn_constants["eye"], self.gdn_constants["tril"],
                    self.gdn_constants["ones"], self.gdn_constants["masks"], recurrent,
                    chunk_size=32, output_final_state=True, compute_kernel_config=self.compute,
                    program_config=ttnn.ChunkGdnFusedProgramConfig(num_receivers=1,
                        **({"num_producers":1} if self.wave_heads else {})),
                    num_heads=min(wave_heads, n * 8 - head_offset), head_offset=head_offset,
                    v_flat=True, HV=8, qk_norm=True, scale=128**-0.5, qk_flat=True, Hk=4,
                    o_out=scratch, final_state_out=recurrent)
            out = ttnn.permute(ttnn.reshape(out, (n, 8, 32, 128))[:, :, :1, :], (0, 2, 1, 3))
            out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
            out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            mixed = self.reduce(self.linear(out, "linear_attn.out_proj"))
            return self.finish(x, mixed)
        finally:
            self._grouped = False
