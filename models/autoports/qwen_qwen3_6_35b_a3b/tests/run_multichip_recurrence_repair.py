# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic native-recurrence candidate; ordinary acceptance gates remain on.

This intentionally tests the native FPU recurrence's different intermediate
rounding and exp(log(decay)) boundary. It is never the default implementation.
"""

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_b1_repair import main
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder


def recurrent(self, q, k, v, g, beta, state, *, decay_value=None, round_state=False, output_state=None):
    if decay_value is not None:
        g = ttnn.log(ttnn.reshape(decay_value, (1, 1, 8)))
    initial = ttnn.unary_chain(state, self.round_widen_chain) if round_state else state
    output, final = ttnn.transformer.fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        ttnn.reshape(beta, (1, 1, 8)),
        initial_state=initial,
        output_final_state=True,
        compute_kernel_config=self.compute,
    )
    final = ttnn.reshape(final, (1, 8, 128, 128))
    if output_state is not None:
        ttnn.copy(final, output_state)
        final = output_state
    return output, final


if __name__ == "__main__":
    MultichipDecoder._recurrent_step = recurrent
    main()
