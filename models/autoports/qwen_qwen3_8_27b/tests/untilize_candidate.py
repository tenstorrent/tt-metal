# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Controlled decode-only untilize epilogue candidate; never the default path."""

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.fused_decoder import FusedDecoder


class UntilizeCandidate(FusedDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        if self.kind == "linear_attention":
            self.weights["linear_attn.untilize_qkv.weight"] = self.weights["linear_attn.packed.weight"][:, :10240]
            self.weights["linear_attn.untilize_tail.weight"] = self.weights["linear_attn.packed.weight"][:, 10240:]
            self.untilize_output = ttnn.zeros(
                [1, 32, 10240],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self

    def _delta(self, x, state):
        b, t, _ = x.shape
        c = self.config
        h, hv, d = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim
        if (b, t) != (1, 1):
            return super()._delta(x, state)
        conv_width = (2 * h + hv) * d
        z_width = hv * d
        gate_width = (hv + 31) // 32 * 32
        tail = self._linear(x, "linear_attn.untilize_tail")
        z = tail[:, :, :z_width]
        beta = ttnn.sigmoid(tail[:, :, z_width : z_width + hv])
        a = ttnn.typecast(tail[:, :, z_width + gate_width : z_width + gate_width + hv], ttnn.float32)
        grid = self.device.compute_with_storage_grid_size()
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            per_core_M=1,
            per_core_N=3,
            fuse_batch=True,
            mcast_in0=True,
            untilize_out=False,
        )
        padded_t = 32
        padded_x = ttnn.pad(x, [(0, 0), (0, 31), (0, 0)], 0.0)
        row_qkv = ttnn.matmul(
            padded_x,
            self.weights["linear_attn.untilize_qkv.weight"],
            dtype=ttnn.bfloat16,
            program_config=config,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_qkv = ttnn.to_layout(row_qkv, ttnn.ROW_MAJOR_LAYOUT)
        chunks = []
        for user in range(b):
            chunks.append(
                ttnn.experimental.kda.qkv_causal_conv1d_silu(
                    row_qkv[user : user + 1],
                    state.conv[user : user + 1],
                    *self.conv_taps,
                    h * d,
                    h * d,
                    hv * d,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                )
            )
        q, k, v = [parts[0] if b == 1 else ttnn.concat(parts, dim=0) for parts in zip(*chunks)]
        history_tail = (
            row_qkv[:, t - 3 : t, :] if t >= 3 else ttnn.concat([state.conv[:, t:, :], row_qkv[:, :t, :]], dim=1)
        )
        ttnn.copy(history_tail, state.conv)

        g = ttnn.mul(
            self.a_neg,
            ttnn.add(a, self.dt_bias),
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        if padded_t != t:

            def pad_time(a):
                return ttnn.pad(a, [(0, 0), (0, padded_t - t), (0, 0)], 0.0)

            # Keep the convolution outputs physically padded. Zero beta and
            # log-decay make every padded recurrence step an identity, even
            # when its convolution Q/K/V values are nonzero.
            g, beta = [pad_time(a) for a in (g, beta)]
        # The native scan assigns one value head to each core. Split only
        # its independent batch axis, preserving the public batch contract.
        grid = self.device.compute_with_storage_grid_size()
        scan_batch = grid.x * grid.y // hv
        outputs, states = [], []
        for start in range(0, b, scan_batch):
            end = min(start + scan_batch, b)
            output_part, state_part = ttnn.transformer.chunk_gated_delta_rule(
                q[start:end],
                k[start:end],
                v[start:end],
                g[start:end],
                beta[start:end],
                initial_state=state.recurrent[start:end],
                output_final_state=True,
                output_head_major=True,
                chunk_size=32,
                **self.delta_constants,
            )
            outputs.append(output_part)
            states.append(state_part)
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)
        new_state = states[0] if len(states) == 1 else ttnn.concat(states, dim=0)
        ttnn.copy(new_state, state.recurrent)
        if padded_t != t:
            z = ttnn.pad(z, [(0, 0), (0, padded_t - t), (0, 0)], 0.0)
        output = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output, z, self.weights["linear_attn.norm.weight"], hv, epsilon=self.eps, output_dtype=ttnn.bfloat16
        )
        output = ttnn.mul(output, z)
        # Hide only trailing rows; retain the identical physical tile geometry.
        # The following projection and norms act independently on each row.
        output = ttnn.reshape(output, [b, t, hv * d], output.padded_shape)
        return self._linear(output, "linear_attn.out_proj")
