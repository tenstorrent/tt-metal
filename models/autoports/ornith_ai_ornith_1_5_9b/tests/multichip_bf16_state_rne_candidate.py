# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Candidate: BF16 persistent pooled state, FP32 HiFi4 recurrence arithmetic."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf16_state_rne_kernel import flat_ordered_post_loop

class BF16State(production.MultichipDecoder):
    def allocate_batch_state(self, **kwargs):
        states = super().allocate_batch_state(**kwargs)
        if self.kind == "linear_attention" and len(states) > 1:
            recurrent = ttnn.typecast(states[0].recurrent, ttnn.bfloat16)
            for state in states:
                state.recurrent = recurrent
        return states

    def read_linear_state(self, state, name):
        value = super().read_linear_state(state, name)
        return ttnn.typecast(value, ttnn.float32) if name == "recurrent" and value.dtype != ttnn.float32 else value

    def write_linear_state(self, state, name, value):
        if name == "recurrent" and state.recurrent.dtype != value.dtype:
            value = ttnn.typecast(value, state.recurrent.dtype)
        super().write_linear_state(state, name, value)

if __name__ == "__main__":
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    production.flat_ordered_post_loop = flat_ordered_post_loop
    stack.MultichipDecoder = BF16State
    stack.main()
