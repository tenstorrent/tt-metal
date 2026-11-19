# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cumulative decoder candidate with flat-input fused recurrent preparation."""
import ttnn
import argparse
import sys
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_kernel_fusion_candidate import KernelFusionCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_recurrent import flat_recurrent
class FlatRecurrentCandidate(KernelFusionCandidate):
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
            old = history[j : j + 1, :, :, :]
            conv = ttnn.addcmul(conv, ttnn.typecast(old, ttnn.float32), self.float_taps[j])
        conv = ttnn.typecast(ttnn.silu(conv), ttnn.bfloat16)
        new_history = ttnn.concat([history[1:, :, :, :], raw], dim=0)
        ttnn.copy(new_history, states[0].conv)

        # The producer initializes every physical tail row before this metadata
        # view exposes it as a neutral GDN token. Public logical rows stay intact.
        recurrent=ttnn.reshape(states[0].recurrent,(n*8,128,128))
        out=flat_recurrent(conv,g,beta,recurrent,self.wave)
        out = ttnn.permute(ttnn.reshape(out, (n, 8, 1, 128)), (0, 2, 1, 3))
        out = ttnn.reshape(self.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm"), (1, 1, n, 1024))
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return out

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--pool-contract',action='store_true')
    args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
    if args.pool_contract:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_pool_contracts as stack
    else:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_tiled_history_stack as stack
    stack.MultichipDecoder=FlatRecurrentCandidate
    stack.main()
if __name__=="__main__":main()
