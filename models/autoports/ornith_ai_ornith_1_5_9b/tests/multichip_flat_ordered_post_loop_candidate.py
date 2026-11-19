# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cumulative tiled convolution and flat-input ordered recurrence."""
import argparse
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_ordered_candidate import FlatOrderedCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_ordered_post_loop import flat_ordered_post_loop as flat_ordered_post
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_tiled_conv import tiled_conv

class FlatOrderedPostLoopCandidate(FlatOrderedCandidate):
    zero_gate_padding = True
    def pooled_linear_attention_decode_batch(self, packed, states):
        """Neutral tail tokens and disjoint head waves update pooled state in place."""
        n = len(states)
        raw = packed[:, :, :, :2048]
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (n, 1, 8), **({"pad_value": 0.0} if self.zero_gate_padding else {}))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.reshape(
            ttnn.multiply(
                ttnn.add(a, self.dt_bias),
                self.decay,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            ),
            (n, 1, 8),
            **({"pad_value": 0.0} if self.zero_gate_padding else {}),
        )
        conv = tiled_conv(raw, states[0].conv, self.float_taps)

        # The producer initializes every physical tail row before this metadata
        # view exposes it as a neutral GDN token. Public logical rows stay intact.
        recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
        return flat_ordered_post(conv,g,beta,recurrent,z,self.weights["linear_attn.norm.weight"],self.config.rms_norm_eps,self.wave)

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--no-gate-padding',action='store_true')
    ap.add_argument('--wave',type=int,default=104)
    ap.add_argument('--fidelity',choices=['HiFi4','HiFi2','LoFi'],default='HiFi4')
    ap.add_argument('--pool-contract',action='store_true')
    ap.add_argument('--direct-probe',action='store_true')
    args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
    FlatOrderedPostLoopCandidate.zero_gate_padding=not args.no_gate_padding
    FlatOrderedPostLoopCandidate.wave=args.wave
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_loop as kernel
    kernel.MATH_FIDELITY=getattr(ttnn.MathFidelity,args.fidelity)
    if args.direct_probe:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_probe as stack
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_loop as kernel
        stack.fused=kernel
        sys.argv += ["--wave",str(args.wave)]
    elif args.pool_contract:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_pool_contracts as stack
    else:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_tiled_history_stack as stack
    stack.MultichipDecoder=FlatOrderedPostLoopCandidate
    stack.main()
if __name__=='__main__':main()
