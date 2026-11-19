# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Legal HiFi4 GDN phased prep/scan family, with <=104 heads per scan."""
import argparse
import json
import sys
from pathlib import Path
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fusion_reference import MultichipDecoder

class PhasedGdnCandidate(MultichipDecoder):
    wave_batch = 13
    use_mcast = True
    force_serial = False
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
        def padded_flat(value, width):
            value = ttnn.reshape(value, (n, 1, width), pad_value=0.0)
            return ttnn.reshape(value, ttnn.Shape([n, 32, width]), value.padded_shape)

        q = padded_flat(conv[:, :, :, :512], 512)
        k = padded_flat(conv[:, :, :, 512:1024], 512)
        v = padded_flat(conv[:, :, :, 1024:], 1024)
        g = ttnn.reshape(g, ttnn.Shape([n, 32, 8]), g.padded_shape)
        beta = ttnn.reshape(beta, ttnn.Shape([n, 32, 8]), beta.padded_shape)
        recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
        outs, final_states = [], []
        for batch_start in range(0, n, self.wave_batch):
            batch_end = min(n, batch_start + self.wave_batch)
            args = ttnn._ttnn.operations.transformer.chunk_gdn_prep(
                q[batch_start:batch_end], k[batch_start:batch_end], v[batch_start:batch_end],
                g[batch_start:batch_end], beta[batch_start:batch_end],
                self.gdn_constants["eye"], self.gdn_constants["tril"],
                self.gdn_constants["ones"], self.gdn_constants["masks"],
                chunk_size=32, compute_kernel_config=self.compute,
                v_flat=True, HV=8, qk_norm=True, scale=128**-0.5,
                qk_flat=True, Hk=4,
            )
            out, final = ttnn._ttnn.operations.transformer.chunk_gdn_scan(
                *args, initial_state=recurrent[batch_start*8:batch_end*8],
                chunk_size=32, output_final_state=True,
                compute_kernel_config=self.compute, use_mcast=self.use_mcast,
                force_serial=self.force_serial,
            )
            outs.append(out)
            final_states.append(final)
        out = ttnn.concat(outs, dim=0) if len(outs) > 1 else outs[0]
        final = ttnn.concat(final_states, dim=0) if len(final_states) > 1 else final_states[0]
        ttnn.copy(ttnn.reshape(final, states[0].recurrent.shape), states[0].recurrent)
        out = ttnn.permute(ttnn.reshape(out, (n, 8, 32, 128))[:, :, :1, :], (0, 2, 1, 3))
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wave-batch", type=int, default=13)
    parser.add_argument("--no-mcast", action="store_true")
    parser.add_argument("--serial-scan", action="store_true")
    args, rest = parser.parse_known_args()
    assert 1 <= args.wave_batch <= 13
    PhasedGdnCandidate.wave_batch = args.wave_batch
    PhasedGdnCandidate.use_mcast = not args.no_mcast
    PhasedGdnCandidate.force_serial = args.serial_scan
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder = PhasedGdnCandidate
    sys.argv = [sys.argv[0], *rest]
    stack.main()
    output = Path(rest[rest.index("--output") + 1])
    data = json.loads(output.read_text()); data["phased_gdn"] = vars(args)
    output.write_text(json.dumps(data, indent=2) + "\n")

if __name__ == "__main__":
    main()
