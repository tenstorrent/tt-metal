# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact raw recurrent primitive with model-owned input layouts and pooled state."""
import argparse
import json
import sys
from pathlib import Path
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_raw_recurrent import raw_recurrent
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fusion_reference import MultichipDecoder

class RawRecurrentCandidate(MultichipDecoder):
    wave = 110
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
        history = states[0].conv
        conv = ttnn.multiply(ttnn.typecast(raw, ttnn.float32), self.float_taps[3])
        for j in range(3):
            old = ttnn.to_layout(ttnn.reshape(history[:, :, j : j + 1, :], (1, 1, n, 2048)), ttnn.TILE_LAYOUT)
            conv = ttnn.addcmul(conv, ttnn.typecast(old, ttnn.float32), self.float_taps[j])
        conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
        new_history = ttnn.concat(
            [history[:, :, 1:, :], ttnn.reshape(ttnn.to_layout(raw, ttnn.ROW_MAJOR_LAYOUT), (n, 1, 1, 2048))], dim=2
        )
        ttnn.copy(new_history, states[0].conv)

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

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wave", type=int, default=110)
    args, rest = parser.parse_known_args()
    RawRecurrentCandidate.wave = args.wave
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder = RawRecurrentCandidate
    sys.argv = [sys.argv[0], *rest]
    stack.main()
    output = Path(rest[rest.index("--output") + 1])
    data = json.loads(output.read_text()); data["raw_recurrent"] = vars(args)
    output.write_text(json.dumps(data, indent=2) + "\n")

if __name__ == "__main__":
    main()
