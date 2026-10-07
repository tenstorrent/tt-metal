# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expert selection, the TTNN form of reference.NomicRouter.

Routing is a discrete decision: an error large enough to reorder two near-tied experts does not
degrade a token's output, it sends the token through a different pair of 4.7M-parameter experts.
That is why this is the one path held at float32, and why the selection happens before the cast
rather than after.
"""

from __future__ import annotations

from functools import cache

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    activation_memory_config,
    block_spread,
    to_device,
    transpose_linear_weight,
)
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import router_program_config
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# ttnn.topk widens a last dim under 64 to 64 before its device op, and for float32 it does so on
# one core: 95 us of the router's 299 at 8x512. Scoring 64 columns in the matmul leaves it nothing
# to pad. The extra columns get PADDING_LOGIT, which softmax turns into exact zeros, so the real
# probabilities and the selection are bit-identical to 8 columns.
SCORED_COLUMNS = 64
PADDING_LOGIT = -1e30

# The most tokens whose router intermediates go to L1: ten small ops, each faster reading and
# writing L1, 44.8 -> 42.9 us a MoE layer at 128 tokens and 49.6 -> 47.1 at 576, bit-identical up
# to 4096. Above, DRAM: the scores alone take 221 KB of every bank at 94k tokens, and
# router_program_config plans the matmul's buffers against the whole of L1, so they clashed. The
# dense weights the experts take follow activation_memory_config instead, and every intermediate is
# freed before the experts run.
_L1_MAX_TOKENS = 4096


def _intermediate_memory(tokens: int) -> ttnn.MemoryConfig:
    return ttnn.L1_MEMORY_CONFIG if tokens <= _L1_MAX_TOKENS else ttnn.DRAM_MEMORY_CONFIG


@cache
def _leading_cores(count: int, grid: ttnn.CoreCoord) -> ttnn.CoreRangeSet:
    return ttnn.num_cores_to_corerangeset(count, grid, row_wise=True)


class TtNomicRouter(LightweightModule):
    """Softmax over all experts in fp32, top-k, then a dense per-expert row of the top-k weights.

    The top-k weights are deliberately NOT renormalized: moe_normalize_expert_weights is false
    in this checkpoint, so they sum to less than 1 and the MoE branch is attenuated relative to
    the residual. Dividing by the top-k sum, which Mixtral and Switch both do and which is the
    reflex to copy, still scores about 0.993 PCC.

    The selection stays in fp32: ttnn.topk accepts it, and only the dense gate is cast, after it.
    Casting the probabilities first instead reroutes 0.34% to 0.59% of tokens, because bfloat16
    quantizes an 8-wide row coarsely enough to turn near-ties into exact ties.

    The softmax needs both halves of the port's compute kernel config. HiFi4 alone measures
    5.4e-3 to 6.8e-3 max-abs against a 5e-3 budget; with fp32 destination accumulation it is
    1.4e-3 to 1.9e-3.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.top_k = config.moe_top_k
        self.num_experts = config.num_experts

        # The one bias-free projection in the model, and the only weight kept in fp32. Its extra
        # columns are zero, so they score exactly 0 before the padding row is added.
        weight = transpose_linear_weight(state_dict[f"{state_dict_prefix}layer.weight"])
        scored = torch.zeros(weight.shape[0], SCORED_COLUMNS, dtype=weight.dtype)
        scored[:, : self.num_experts] = weight
        padding = torch.full((1, 1, 1, SCORED_COLUMNS), PADDING_LOGIT)
        padding[..., : self.num_experts] = 0.0
        self.weight = to_device(scored, device, dtype=tt_config.matmul_weight_dtype(OpGroup.ROUTER))
        self.padding = to_device(padding, device, dtype=tt_config.router_dtype)

        # 0/1 matrices for dense_weights: spread copies selected index k across column block k,
        # fold sums the K blocks back onto E columns. expert_ids holds 0..E-1 in every block.
        experts, top_k = self.num_experts, self.top_k
        spread = block_spread(top_k, experts)
        fold = torch.cat([torch.eye(experts)] * top_k)
        expert_ids = torch.arange(experts, dtype=torch.float32).repeat(top_k).reshape(1, 1, 1, top_k * experts)
        self.spread, self.fold, self.expert_ids = (
            to_device(tensor, device, dtype=tt_config.activation_dtype) for tensor in (spread, fold, expert_ids)
        )

    def _tile_cores(self, tensor: ttnn.Tensor) -> ttnn.CoreRangeSet | None:
        """One core a tile of a (1, 1, T, C) output, or None (the whole grid) once there are as many tiles.

        A binary op spreads its tiles over the whole grid by default, and its enqueue cost grows with
        the cores it sets up: 23 us on 110 cores against 7 on 4 to 8 for each of the three below at
        128 tokens, in the same 2 us on device, where a core's one tile is the critical path either
        way. Bit-identical: the tiles are computed alike wherever they land.
        """
        grid = self.tt_config.core_grid
        tiles = ttnn.core.divup(tensor.shape[-2], ttnn.TILE_SIZE) * ttnn.core.divup(tensor.shape[-1], ttnn.TILE_SIZE)
        return None if tiles >= grid.x * grid.y else _leading_cores(tiles, grid)

    def logits(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, T, H) -> (1, 1, T, SCORED_COLUMNS) float32 scores, the experts first.

        The bfloat16 activation meets the fp32 weight inside the matmul. Casting it to fp32 first
        measured the same error to 16 digits, at twice the read plus the cast.
        """
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.ROUTER)
        memory = _intermediate_memory(x.shape[-2])
        scores = ttnn.linear(
            x,
            self.weight,
            dtype=self.tt_config.router_dtype,
            program_config=router_program_config(
                x.shape[-2],
                x.shape[-1] // ttnn.TILE_SIZE,
                SCORED_COLUMNS // ttnn.TILE_SIZE,
                x.dtype,
                self.weight.dtype,
                self.tt_config.router_dtype,
                self.tt_config.core_grid,
                self.tt_config.l1_cb_bytes,
                compute_kernel_config,
            ),
            compute_kernel_config=compute_kernel_config,
            memory_config=memory,
        )
        # Its own fp32 add, exact on every column: given as the matmul's fused bias, the padding row
        # rounded all 64 outputs, up to 3e-3 on the real logits.
        logits = ttnn.add(
            scores,
            self.padding,
            dtype=self.tt_config.router_dtype,
            memory_config=memory,
            sub_core_grids=self._tile_cores(scores),
        )
        ttnn.deallocate(scores)
        return logits

    def select(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Score every expert and take the top-k, all in float32.

        Split out of forward so the per-module test can assert on the selection itself: index
        agreement against torch is the gate, and PCC on the dense output cannot express it.

        Args:
            x: (1, 1, T, H) flat token activations.

        Returns:
            tuple: probabilities (1, 1, T, E) fp32, values (1, 1, T, K) fp32 unrenormalized, and
            indices (1, 1, T, K) uint32.
        """
        logits = self.logits(x)
        memory = _intermediate_memory(x.shape[-2])
        scored = ttnn.softmax(
            logits,
            dim=-1,
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.SOFTMAX),
            memory_config=memory,
        )
        ttnn.deallocate(logits)
        values, indices = ttnn.topk(scored, k=self.top_k, dim=-1, memory_config=memory)
        probabilities = ttnn.slice(
            scored, [0, 0, 0, 0], [1, 1, scored.shape[-2], self.num_experts], memory_config=memory
        )
        ttnn.deallocate(scored)
        return probabilities, values, indices

    def dense_weights(self, probabilities: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, T, E) probabilities and (1, 1, T, K) top-k indices -> (1, 1, T, E) bfloat16 gate.

        What ttnn.scatter of the top-k values computes, bit for bit, in 27 us against its 126 a MoE
        layer at 8x512: scatter untilizes its three inputs and tilizes its output. The indices are
        compared as one-hots against 0..E-1 and the probabilities multiplied by the mask, so no
        column of the (T, K) top-k output is sliced out; ttnn.slice does that through row-major, 49 us
        a column. Both matmuls take a 0/1 matrix against small integers or 0/1 values, one non-zero
        term to a sum, so they are exact; the selected indices are distinct, so the mask is 0/1.
        """
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.ROUTER)
        activation_dtype = self.tt_config.activation_dtype
        memory = _intermediate_memory(probabilities.shape[-2])
        ids = ttnn.typecast(indices, activation_dtype, memory_config=memory)
        spread = ttnn.matmul(
            ids, self.spread, dtype=activation_dtype, compute_kernel_config=compute_kernel_config, memory_config=memory
        )
        ttnn.deallocate(ids)
        hits = ttnn.eq(spread, self.expert_ids, memory_config=memory, sub_core_grids=self._tile_cores(spread))
        ttnn.deallocate(spread)
        mask = ttnn.matmul(
            hits, self.fold, dtype=activation_dtype, compute_kernel_config=compute_kernel_config, memory_config=memory
        )
        ttnn.deallocate(hits)
        # The fp32 product written straight to bfloat16: bit-identical to an fp32 product and a
        # typecast, at 128 to 4096 tokens, and one op fewer.
        dense = ttnn.multiply(
            probabilities,
            mask,
            dtype=activation_dtype,
            memory_config=activation_memory_config(probabilities),
            sub_core_grids=self._tile_cores(probabilities),
        )
        ttnn.deallocate(mask)
        return dense

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Produce the dense routing weights the expert gate multiply consumes.

        Args:
            x: (1, 1, T, H) flat token activations.

        Returns:
            ttnn.Tensor: (1, 1, T, E) in the activation dtype, carrying the routed weight at the
            top-k positions and zero everywhere else.
        """
        probabilities, values, indices = self.select(x)
        ttnn.deallocate(values)
        dense = self.dense_weights(probabilities, indices)
        ttnn.deallocate(probabilities)
        ttnn.deallocate(indices)
        return dense
