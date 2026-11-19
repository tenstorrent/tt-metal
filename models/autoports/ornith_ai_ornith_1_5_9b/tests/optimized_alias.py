# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode fused GDN state-output alias candidate; each head owns disjoint state."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_movement import MovementCandidate


class AliasCandidate(MovementCandidate):
    alias_gdn = False

    def linear_attention(self, x, state, *, decode):
        n = x.shape[2]
        packed = self.linear(x, "packed_input")
        z = packed[:, :, :, 8192:12288]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 12288:12320]), (1, n, 32))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 12320:12352], (1, n, 32)), ttnn.float32)
        g = ttnn.multiply(
            ttnn.add(a, self.dt_bias),
            self.decay,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        old_history = ttnn.reshape(ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT), (1, 3, 8192))
        padded = packed
        if n % 32:
            padded = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(padded, (1, padded.shape[2], 12352)),
            old_history,
            *self.taps,
            2048,
            2048,
            4096,
            actual_start=self.conv_start,
            predecessor_carry=old_history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self.conv_channels),
            compute_kernel_config=self.conv_compute,
        )
        if n >= 3:
            ttnn.copy(packed[:, :, n - 3 : n, :8192], state.conv)
        else:
            history = ttnn.concat([state.conv, packed[:, :, :, :8192]], dim=2)
            ttnn.copy(history[:, :, n : n + 3, :], state.conv)

        if n % 32:
            # Inactive tail rows leave recurrence unchanged: exp(g)=1 and beta=0.
            g = ttnn.pad(g, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        if self.alias_gdn and decode:
            recurrent = ttnn.reshape(state.recurrent, (32, 128, 128))
            scratch = ttnn.empty(
                (32, 1, 32, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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
                program_config=ttnn.ChunkGdnFusedProgramConfig(num_receivers=self.gdn_receivers),
                v_flat=True,
                HV=32,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=16,
                o_out=scratch,
                final_state_out=recurrent,
            )
            out = ttnn.reshape(out, (32, 32, 128))
        else:
            out, new_state = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state.recurrent,
                output_final_state=True,
                compute_kernel_config=self.compute,
                program_config=self.gdn_program(n),
                chunk_size=32,
                num_key_heads=16,
                output_head_major=True,
                **self.gdn_constants,
            )
            ttnn.copy(new_state, state.recurrent)

        if self.fused_gated_norm:
            out = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                ttnn.reshape(ttnn.typecast(out, ttnn.bfloat16), (32, padded.shape[2], 128)),
                ttnn.reshape(padded, (1, padded.shape[2], 12352)),
                ttnn.reshape(self.weights["linear_attn.norm.weight"], (128,)),
                32,
                epsilon=self.config.rms_norm_eps,
                output_dtype=ttnn.bfloat16,
                gate_column_offset=8192,
                compute_kernel_config=self.conv_compute,
            )
            out = ttnn.reshape(out, (1, 1, padded.shape[2], 4096))[:, :, :n, :]
            return self.linear(ttnn.multiply(z, out), "linear_attn.out_proj")
        out = ttnn.typecast(out, ttnn.bfloat16)
        out = self.norm(out, "linear_attn.norm")
        out = ttnn.transformer.concatenate_heads(ttnn.reshape(out, (1, 32, padded.shape[2], 128)))
        out = ttnn.reshape(out, (1, 1, padded.shape[2], 4096))[:, :, :n, :]
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return self.linear(out, "linear_attn.out_proj")
