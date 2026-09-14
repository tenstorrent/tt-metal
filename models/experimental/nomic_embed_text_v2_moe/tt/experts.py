# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The expert FFN bank, the TTNN form of reference.NomicExperts.dense_forward.

Upstream gathers each expert's tokens by value, runs them, and scatter-adds the results back.
That loop is data-dependent, so it has no device form. dense_forward runs every token through
every expert instead and zeroes the unrouted contributions with the gate, which is
arithmetically identical and reduces to two broadcast-batch matmuls, a multiply and a reduce.

    x            (1, 1, T, H)
      matmul w1  (1, E, T, F)   every token through every expert
      gelu, w2   (1, E, T, H)
      gate       (1, E, T, 1)   from the router, zero off the top-k
      reduce E   (1, 1, T, H)
      + bias     one shared (H,) vector, once, after the sum

The (1, E, T, F) intermediate is the port's peak transient, about 50 MB at T=1024 in bfloat16,
and it scales with batch times sequence length rather than sequence length alone.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import pack_expert_weights, to_device


class TtNomicExperts(LightweightModule):
    """All eight experts in two packed operands, plus one bias shared across them.

    The bias is added once after the weighted sum. Adding it inside the per-expert loop scales
    it by the routed-weight sum, leaving an offset of (sum(w) - 1) * bias, which is real because
    the weights are not renormalized. That offset is nearly constant across tokens and PCC
    mean-centres, so the wrong version still scores 0.9999998; only max-abs sees it.

    pack_expert_weights is what makes the w2 misorientation loud. Viewing it as (E, H, F) rather
    than (E, F, H) is an equally legal reshape, since E*F*H is symmetric in those two, and in
    torch the wrong slab plus a .T typechecks and returns noise. As a 4D operand there is no .T
    to paper over it and the matmul's inner dimensions stop agreeing.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config

        w1, w2 = pack_expert_weights(
            state_dict[f"{state_dict_prefix}mlp.w1"], state_dict[f"{state_dict_prefix}mlp.w2"], config
        )
        self.w1 = to_device(w1, device, dtype=tt_config.weight_dtype)
        self.w2 = to_device(w2, device, dtype=tt_config.weight_dtype)
        self.bias = to_device(
            state_dict[f"{state_dict_prefix}bias"].reshape(1, 1, 1, config.hidden_size),
            device,
            dtype=tt_config.weight_dtype,
        )

    def forward(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run every expert, weight the outputs by the routing, and sum them.

        Args:
            x: (1, 1, T, H) flat token activations.
            dense_weights: (1, 1, T, E) from TtNomicRouter, zero off the top-k.

        Returns:
            ttnn.Tensor: (1, 1, T, H).
        """
        tokens, hidden = x.shape[-2], x.shape[-1]

        hidden_states = ttnn.matmul(x, self.w1, compute_kernel_config=self.tt_config.compute_kernel_config)
        activated = ttnn.gelu(hidden_states)
        ttnn.deallocate(hidden_states)

        per_expert = ttnn.matmul(activated, self.w2, compute_kernel_config=self.tt_config.compute_kernel_config)
        ttnn.deallocate(activated)

        # (1, 1, T, E) -> (1, E, T, 1), whose trailing singleton broadcasts over the hidden axis.
        gate = ttnn.permute(dense_weights, (0, 3, 2, 1))
        gated = ttnn.multiply(per_expert, gate)
        ttnn.deallocate(per_expert)
        ttnn.deallocate(gate)

        summed = ttnn.experimental.fast_reduce_nc(
            gated, dims=[1], compute_kernel_config=self.tt_config.compute_kernel_config
        )
        ttnn.deallocate(gated)

        # fast_reduce_nc returns the tile-padded row count, not T: at T=74 it reports 96 rows
        # with the trailing 22 zero. The data is right, the logical shape is not.
        if summed.shape[-2] != tokens:
            trimmed = ttnn.slice(summed, [0, 0, 0, 0], [1, 1, tokens, hidden])
            ttnn.deallocate(summed)
            summed = trimmed

        out = ttnn.add(summed, self.bias)
        ttnn.deallocate(summed)
        return out
