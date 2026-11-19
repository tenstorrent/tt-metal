# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Own neutral GDN tails with a setup mask instead of repeated tiny pad kernels."""

import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder, DecoderState


class DecodeMaskCandidate(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        self = super().from_state_dict(sd, **kwargs)
        mask = torch.zeros(1, 32, 8)
        mask[:, 0] = 1
        self.active_decode_row = ttnn.from_torch(
            mask,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return self

    def linear_attention(self, x, state, *, decode, packed=None, project=True):
        row = getattr(state, "state_row", None)
        if row is not None:
            local = DecoderState(conv=state.conv[row : row + 1], recurrent=state.recurrent[row : row + 1])
            out = self.linear_attention(x, local, decode=decode, packed=packed, project=project)
            self.write_linear_state(state, "conv", local.conv)
            self.write_linear_state(state, "recurrent", local.recurrent)
            return out
        n = x.shape[2]
        packed = self.linear(x, "packed_input") if packed is None else packed
        z = packed[:, :, :, 2048:3072]
        source = ttnn.pad(packed, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0) if decode else packed
        gate_rows = 32 if decode else n
        beta = ttnn.reshape(ttnn.sigmoid(source[:, :, :, 3072:3080]), (1, gate_rows, 8))
        a = ttnn.typecast(ttnn.reshape(source[:, :, :, 3104:3112], (1, gate_rows, 8)), ttnn.float32)
        g = ttnn.multiply(
            ttnn.add(a, self.dt_bias),
            self.decay,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        old_history = ttnn.reshape(ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT), (1, 3, 2048))
        if decode:
            g = ttnn.multiply(g, self.active_decode_row)
            beta = ttnn.multiply(beta, self.active_decode_row)
        padded = source
        if n % 32 and not decode:
            padded = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(padded, (1, padded.shape[2], 3136)),
            old_history,
            *self.taps,
            512,
            512,
            1024,
            actual_start=self.conv_start,
            predecessor_carry=old_history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
            compute_kernel_config=self.conv_compute,
        )
        # Keep the three logical history rows in row-major storage. Reuse the
        # convolution view instead of a concat that tilizes and immediately untilizes.
        if n >= 3:
            history = ttnn.to_layout(packed[:, :, n - 3 : n, :2048], ttnn.ROW_MAJOR_LAYOUT)
        else:
            history = ttnn.concat(
                [
                    ttnn.reshape(old_history, (1, 1, 3, 2048)),
                    ttnn.untilize_with_unpadding(packed, (0, 0, n - 1, 2047)),
                ],
                dim=2,
            )
            ttnn.slice(history, (0, 0, n, 0), (1, 1, n + 3, 2048), (1, 1, 1, 1), output_tensor=state.conv)
        if n >= 3:
            ttnn.copy(history, state.conv)

        if n % 32 and not decode:
            # Inactive tail rows leave recurrence unchanged: exp(g)=1 and beta=0.
            g = ttnn.pad(g, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        if decode:
            # Each receiver reads its complete, disjoint head/V state slab before
            # publishing the final slab. The initial/final views share backing storage;
            # scratch is consumed within this forward and never escapes a live trace.
            recurrent = ttnn.reshape(state.recurrent, (8, 128, 128))
            scratch = ttnn.empty(
                (8, 1, 32, 128),
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
                program_config=ttnn.ChunkGdnFusedProgramConfig(num_receivers=1),
                v_flat=True,
                HV=8,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=4,
                o_out=scratch,
                final_state_out=recurrent,
            )
            out = ttnn.reshape(out, (8, 32, 128))
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
                chunk_size=32,
                num_key_heads=4,
                output_head_major=True,
                **self.gdn_constants,
            )
            ttnn.copy(new_state, state.recurrent)

        out = ttnn.typecast(out, ttnn.bfloat16)
        out = self.norm(out, "linear_attn.norm")
        out = ttnn.transformer.concatenate_heads(ttnn.reshape(out, (1, 8, padded.shape[2], 128)))
        out = ttnn.reshape(out, (1, 1, padded.shape[2], 1024))[:, :, :n, :]
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return self.reduce(self.linear(out, "linear_attn.out_proj")) if project else out
