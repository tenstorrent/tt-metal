# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated gated RMSNorm and head concatenation over the GDN output."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_create_candidate import QkvCreateCandidate

class GatedNormCandidate(QkvCreateCandidate):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        if self.kind == "linear_attention":
            self.gated_norm_weight = ttnn.reshape(self.weights["linear_attn.norm.weight"], (128,))
            self.gated_norm_compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=False,
            )
        return self

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
            old = ttnn.to_layout(ttnn.reshape(history[:, :, j : j + 1, :], (1, 1, n, 2048)), ttnn.TILE_LAYOUT)
            conv = (ttnn.addcmul(conv, ttnn.typecast(old, ttnn.float32), self.float_taps[j])
                    if self.fused_conv else ttnn.add(conv, ttnn.multiply(ttnn.typecast(old, ttnn.float32), self.float_taps[j])))
        conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
        new_history = ttnn.concat(
            [history[:, :, 1:, :], ttnn.reshape(ttnn.to_layout(raw, ttnn.ROW_MAJOR_LAYOUT), (n, 1, 1, 2048))], dim=2
        )
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
        gates = ttnn.reshape(z, (n, 1, 1024), pad_value=0.0)
        gates = ttnn.reshape(gates, ttnn.Shape([n, 32, 1024]), gates.padded_shape)
        normalized = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            ttnn.typecast(ttnn.reshape(out, (n * 8, 32, 128)), ttnn.bfloat16),
            gates, self.gated_norm_weight, 8,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.gated_norm_compute,
            output_dtype=ttnn.bfloat16,
        )
        normalized = ttnn.reshape(normalized[:, :1, :], (1, 1, n, 1024))
        # Dedicated op supplies sigmoid(z); multiplying by z completes SiLU.
        return ttnn.multiply(z, normalized)


def main():
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder = GatedNormCandidate
    stack.main()

if __name__ == "__main__":
    main()
