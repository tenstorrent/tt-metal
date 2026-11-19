# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Conditional cumulative precise conv-wide gates plus NoC zero-fill reader."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fast_zero_conv_gates_wide_modes import conv_gates_precise
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fast_zero_conv_gates_recurrent import flat_ordered_conv_gates


class FastZeroConvGatesCandidate(MultichipDecoder):
    def pooled_linear_attention_decode_batch(self, packed, states):
        n = len(states)
        conv, g, beta = conv_gates_precise(packed, states[0].conv, self.float_taps, self.dt_bias, self.decay)
        recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
        return flat_ordered_conv_gates(
            conv, g, beta, recurrent, packed,
            self.weights["linear_attn.norm.weight"], self.config.rms_norm_eps, 104,
        )


if __name__ == "__main__":
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder = FastZeroConvGatesCandidate
    stack.main()
