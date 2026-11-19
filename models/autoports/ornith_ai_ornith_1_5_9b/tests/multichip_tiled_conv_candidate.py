# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct tiled convolution crossed with the recurrent implementation family."""
import argparse
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_tiled_conv import tiled_conv
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_recurrent_candidate import FlatRecurrentCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_recurrent import flat_recurrent
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_kernel_fusion_candidate import KernelFusionCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_raw_recurrent import raw_recurrent


class RawTiledConvCandidate(KernelFusionCandidate):
    def pooled_linear_attention_decode_batch(self, packed, states):
        """Neutral tail tokens and disjoint head waves update pooled state in place."""
        n = len(states)
        if not self.use_raw:
            return super().pooled_linear_attention_decode_batch(packed, states)
        raw = packed[:, :, :, :2048]
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8), pad_value=0.0)
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.reshape(
            ttnn.multiply(
                ttnn.add(a, self.dt_bias),
                self.decay,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            ),
            (n, 1, 8),
            pad_value=0.0,
        )
        conv = tiled_conv(raw, states[0].conv, self.float_taps)

        # The producer initializes every physical tail row before this metadata
        # view exposes it as a neutral GDN token. Public logical rows stay intact.
        q = ttnn.typecast(ttnn.reshape(conv[:, :, :, :512], (n, 1, 4, 128)), ttnn.float32)
        k = ttnn.typecast(ttnn.reshape(conv[:, :, :, 512:1024], (n, 1, 4, 128)), ttnn.float32)
        v = ttnn.typecast(ttnn.reshape(conv[:, :, :, 1024:], (n, 1, 8, 128)), ttnn.float32)
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6 / 128, compute_kernel_config=self.compute), 1/128)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6 / 128, compute_kernel_config=self.compute), 128**-0.5)
        def heads(x, h):
            return ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (n*h, 1, 128))
        q = ttnn.repeat_interleave(heads(q, 4), 2, 0)
        k = ttnn.repeat_interleave(heads(k, 4), 2, 0)
        v = heads(v, 8)
        def scalar_heads(x):
            x = ttnn.to_layout(ttnn.permute(x, (0, 2, 1)), ttnn.ROW_MAJOR_LAYOUT)
            return ttnn.to_layout(ttnn.reshape(x, (n*8, 1, 1)), ttnn.TILE_LAYOUT)
        decay = ttnn.exp(scalar_heads(g))
        beta = scalar_heads(ttnn.typecast(beta, ttnn.float32))
        recurrent = ttnn.reshape(states[0].recurrent, (n*8, 128, 128))
        out = raw_recurrent(q, k, v, decay, beta, recurrent, self.wave)
        out = ttnn.permute(ttnn.reshape(out, (n, 8, 1, 128)), (0, 2, 1, 3))
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out


class FlatTiledConvCandidate(FlatRecurrentCandidate):
    def pooled_linear_attention_decode_batch(self, packed, states):
        """Neutral tail tokens and disjoint head waves update pooled state in place."""
        n = len(states)
        raw = packed[:, :, :, :2048]
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8), pad_value=0.0)
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.reshape(
            ttnn.multiply(
                ttnn.add(a, self.dt_bias),
                self.decay,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            ),
            (n, 1, 8),
            pad_value=0.0,
        )
        conv = tiled_conv(raw, states[0].conv, self.float_taps)

        # The producer initializes every physical tail row before this metadata
        # view exposes it as a neutral GDN token. Public logical rows stay intact.
        recurrent=ttnn.reshape(states[0].recurrent,(n*8,128,128))
        out=flat_recurrent(conv,g,beta,recurrent,self.wave)
        out = ttnn.permute(ttnn.reshape(out, (n, 8, 1, 128)), (0, 2, 1, 3))
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out


class ChunkTiledConvCandidate(KernelFusionCandidate):
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
        conv = tiled_conv(raw, states[0].conv, self.float_taps)

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


def main():
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--recurrence',choices=['raw','flat','chunk'],default='chunk');ap.add_argument('--pool-contract',action='store_true');args,rest=ap.parse_known_args()
    if args.pool_contract:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_pool_contracts as stack
    else:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_tiled_history_stack as stack
    stack.MultichipDecoder={'raw':RawTiledConvCandidate,'flat':FlatTiledConvCandidate,'chunk':ChunkTiledConvCandidate}[args.recurrence]
    sys.argv=[sys.argv[0],*rest];stack.main()
if __name__=='__main__':main()
