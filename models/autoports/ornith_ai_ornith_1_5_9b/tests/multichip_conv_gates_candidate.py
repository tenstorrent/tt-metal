# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepare all batch gates once in the convolution launch, then ordered recurrence."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_loop_candidate as base
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_conv_gates import conv_gates
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_ordered_conv_gates import flat_ordered_conv_gates

class ConvGatesCandidate(base.FlatOrderedPostLoopCandidate):
    def pooled_linear_attention_decode_batch(self, packed, states):
        n=len(states)
        conv,g,beta=conv_gates(packed,states[0].conv,self.float_taps,self.dt_bias,self.decay)
        recurrent=ttnn.reshape(states[0].recurrent,(n*8,128,128))
        return flat_ordered_conv_gates(conv,g,beta,recurrent,packed,
            self.weights["linear_attn.norm.weight"],self.config.rms_norm_eps,self.wave)

if __name__ == "__main__":
    base.FlatOrderedPostLoopCandidate=ConvGatesCandidate
    base.main()
