# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vectorized independent linear-attention requests using the fused recurrent op."""

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batched_candidate import BatchedCandidate


class RecurrentCandidate(BatchedCandidate):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        if self.kind == "linear_attention":
            self.float_taps = [ttnn.typecast(t, ttnn.float32) for t in self.taps]
        return self

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        if self.kind != "linear_attention":
            return super().decode_batch_packed(
                x, states=states, current_positions=current_positions, page_tables=page_tables
            )
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
            history = ttnn.concat([st.conv for st in states], dim=0) if n > 1 else states[0].conv
            conv = ttnn.multiply(ttnn.typecast(raw, ttnn.float32), self.float_taps[3])
            for j in range(3):
                old = ttnn.to_layout(ttnn.reshape(history[:, :, j : j + 1, :], (1, 1, n, 2048)), ttnn.TILE_LAYOUT)
                conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(old, ttnn.float32), self.float_taps[j]))
            conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
            new_history = ttnn.concat(
                [history[:, :, 1:, :], ttnn.reshape(ttnn.to_layout(raw, ttnn.ROW_MAJOR_LAYOUT), (n, 1, 1, 2048))], dim=2
            )
            for i, st in enumerate(states):
                ttnn.copy(new_history[i : i + 1, :, :, :], st.conv)
            q = ttnn.typecast(ttnn.reshape(conv[:, :, :, :512], (n, 1, 4, 128)), ttnn.float32)
            k = ttnn.typecast(ttnn.reshape(conv[:, :, :, 512:1024], (n, 1, 4, 128)), ttnn.float32)
            v = ttnn.reshape(conv[:, :, :, 1024:], (n, 1, 8, 128))
            q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6 / 128, compute_kernel_config=self.compute), 128**-0.5)
            k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6 / 128, compute_kernel_config=self.compute), 128**-0.5)
            recurrent = ttnn.concat([st.recurrent for st in states], dim=0) if n > 1 else states[0].recurrent
            parts, final_parts = [], []
            # One core per value head: twelve requests use96 of110 available cores.
            for start in range(0, n, 12):
                end = min(start + 12, n)
                part, last = ttnn.transformer.fused_recurrent_gated_delta_rule(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    g[start:end],
                    beta[start:end],
                    initial_state=recurrent[start:end],
                    output_final_state=True,
                    compute_kernel_config=self.compute,
                )
                parts.append(part)
                final_parts.append(last)
            out = ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]
            final = ttnn.concat(final_parts, dim=0) if len(final_parts) > 1 else final_parts[0]
            for i, st in enumerate(states):
                ttnn.copy(final[i : i + 1, :, :, :], st.recurrent)
            out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
            out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            mixed = self.reduce(self.linear(out, "linear_attn.out_proj"))
            return self.finish(x, mixed)
        finally:
            self._grouped = False
