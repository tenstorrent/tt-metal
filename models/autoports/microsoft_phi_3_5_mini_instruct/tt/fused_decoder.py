# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused, single-device Phi-3.5 decoder layer.

The public prefill, decode, and paged-cache contracts are inherited unchanged
from :class:`FunctionalDecoder`.  Runtime dispatch reaches the override below:
the packed gate/up projection is still split without host movement, while SiLU
is evaluated inside the consuming multiply kernel instead of as a standalone
unary operation.
"""

from __future__ import annotations

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder


class FusedDecoder(FunctionalDecoder):
    """Phi-3.5 decoder with correctness- and latency-proven graph rewrites."""

    def _mlp(self, hidden_states):
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        gate_up = ttnn.linear(normalized, self.weights["gate_up"], dtype=ttnn.bfloat16)
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        activated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.add(hidden_states, ttnn.linear(activated, self.weights["down"], dtype=ttnn.bfloat16))
