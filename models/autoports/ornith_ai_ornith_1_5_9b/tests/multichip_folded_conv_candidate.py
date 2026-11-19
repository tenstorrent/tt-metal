# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent decode users folded into depthwise channels for fused convolution."""

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


class FoldedConvCandidate(MultichipDecoder):
    channel_chunk = 128

    def allocate_batch_state(self, *, batch, context=None):
        states = super().allocate_batch_state(batch=batch, context=context)
        if self.kind == "linear_attention" and batch > 1:
            if not hasattr(self, "folded_taps"):
                self.folded_taps = {}
            self.folded_taps[batch] = [
                ttnn.pad(ttnn.repeat(ttnn.reshape(t, (1, 1, 2048)), (1, 1, batch)), [(0, 0), (0, 0), (0, self.channel_chunk)], 0.0)
                for t in self.taps
            ]
        return states

    def pooled_linear_attention_decode_batch(self, packed, states):
        """Neutral tail tokens and disjoint head waves update pooled state in place."""
        n = len(states)
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
        folded = ttnn.reshape(raw, (1, 1, n * 2048))
        folded = ttnn.pad(folded, [(0, 0), (0, 31), (0, self.channel_chunk)], 0.0)
        old = ttnn.reshape(ttnn.permute(history, (1, 2, 0, 3)), (1, 3, n * 2048))
        old = ttnn.pad(old, [(0, 0), (0, 0), (0, self.channel_chunk)], 0.0)
        convolved, unused_k, unused_v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            folded,
            old,
            *self.folded_taps[n],
            n * 2048,
            32,
            self.channel_chunk - 32,
            actual_start=self.conv_start,
            predecessor_carry=old,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self.channel_chunk),
            compute_kernel_config=self.conv_compute,
        )
        conv = ttnn.reshape(convolved[:, :1, :], (1, 1, n, 2048))
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
