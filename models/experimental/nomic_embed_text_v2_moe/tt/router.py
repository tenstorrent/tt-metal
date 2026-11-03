# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expert selection, the TTNN form of reference.NomicRouter.

Routing is a discrete decision: an error large enough to reorder two near-tied experts does not
degrade a token's output, it sends the token through a different pair of 4.7M-parameter experts.
That is why this is the one path held at float32, and why the selection happens before the cast
rather than after.
"""

from __future__ import annotations

from typing import NamedTuple

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device, transpose_linear_weight
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# The most tokens whose scores go to L1: the ops after the router, routing_metadata first, read them faster
# there, and above it the scores take 221 KB of every bank at 94k tokens.
_L1_MAX_TOKENS = 4096


def _scores_memory(tokens: int) -> ttnn.MemoryConfig:
    return ttnn.L1_MEMORY_CONFIG if tokens <= _L1_MAX_TOKENS else ttnn.DRAM_MEMORY_CONFIG


class Routing(NamedTuple):
    """The router's scores, (1, 1, T, E) float32 probabilities, E the experts. routing_metadata selects
    each token's top-k from them, the weights unrenormalized. Up to 4096 tokens they sit in L1."""

    scores: ttnn.Tensor

    def deallocate(self) -> None:
        for tensor in self:
            ttnn.deallocate(tensor)


class TtNomicRouter(LightweightModule):
    """Softmax over all experts in fp32, from which routing_metadata selects the top-k, and the tokens' rows
    row-major for the routed dispatch, in one op (router_front_end).

    The top-k weights are deliberately NOT renormalized: moe_normalize_expert_weights is false
    in this checkpoint, so they sum to less than 1 and the MoE branch is attenuated relative to
    the residual. Dividing by the top-k sum, which Mixtral and Switch both do and which is the
    reflex to copy, still scores about 0.993 PCC.

    The selection stays in fp32: routing_metadata selects from the float32 probabilities, as ttnn.topk
    did before it, and copies the selected weights as they are. Casting the probabilities first
    instead reroutes 0.34% to 0.59% of tokens, because bfloat16 quantizes an 8-wide row coarsely
    enough to turn near-ties into exact ties.

    The softmax needs both halves of the router's compute kernel config. HiFi4 alone measures
    5.4e-3 to 6.8e-3 max-abs against a 5e-3 budget; with fp32 destination accumulation it is
    1.4e-3 to 1.9e-3.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.num_experts = config.num_experts

        # The one bias-free projection in the model, and the only weight kept in fp32. E columns wide: the
        # tile's padded columns are zero, and the softmax masks them out (forward).
        weight = transpose_linear_weight(state_dict[f"{state_dict_prefix}layer.weight"])
        self.weight = to_device(weight, device, dtype=tt_config.matmul_weight_dtype(OpGroup.ROUTER))

    def forward(
        self, x: ttnn.Tensor, rows_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> tuple[Routing, ttnn.Tensor]:
        """Score every expert, in float32, and lay x's rows out row-major for the dispatch, in one pass over x.

        router_front_end reads x in whatever layout it comes, the split norm's or interleaved. Its matmul takes
        K in one block, accumulated in the fp32 DST, and its softmax runs ttnn.softmax's numerically stable
        steps with the -inf padding mask for E columns of a tile, so the probabilities are, to the bit, those
        of ttnn.linear with K in one block and ttnn.softmax; the rows are x untilized. At 8x512 it took the
        place of a copy of x for the router, the router matmul, the softmax and the dispatch table's untilize.

        Args:
            x: (1, 1, T, H) flat token activations.
            rows_memory_config: where the rows go (TtNomicExperts.dispatch_table_config).

        Returns:
            Routing: probabilities (1, 1, T, E) fp32; and the rows, (1, 1, T, H) bfloat16 row-major.
        """
        scores, rows = ttnn.experimental.nomic_embed_text_v2_moe.router_front_end(
            x,
            self.weight,
            scores_memory_config=_scores_memory(x.shape[-2]),
            rows_memory_config=rows_memory_config,
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.ROUTER),
        )
        return Routing(scores=scores), rows
