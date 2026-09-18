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

T above is the whole batch's token count, so it is the one axis here that grows without bound,
and past one output tile row per core the broadcast-batch matmul deadlocks in ttnn. The
pipeline is therefore run in passes over the token axis and the results concatenated; see the
constant below.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import pack_expert_weights, to_device

# ttnn.matmul deadlocks, rather than raising, once a broadcast-batch matmul needs more than one
# output tile row per core. Keeping per_core_M at 1 is what this constant enforces.
#
# The bug is upstream, on the path an (in0_B == 1, in1_B > 1) matmul is auto-routed to.
# reader_bmm_tile_layout_in0_sender_padding.cpp keeps in0 resident in L1 across weight batches
# and replays it by pushing num_blocks_inner_dim blocks per extra batch, while
# bmm_large_block_zm_fused_bias_activation.cpp consumes
# num_blocks_h_dim * num_blocks_w_dim * num_blocks_inner_dim of them per batch. The two agree
# only while both of those outer counts are 1; otherwise the reader under-produces and every
# core blocks forever in cb_wait_front on in0. per_core_M stepping to 2 pushes the L1 estimate
# over budget, which halves out_block_w and makes num_blocks_w_dim 2, which exposes it.
#
# Measured on this p300c, grid 11x10 = 110 cores, no program config and no core grid passed,
# which is what this port does: (1, 1, T, 768) x (1, E, 768, 3072) returns in under a second up
# to T = 3520 (110 tiles, per_core_M 1) and never returns at T = 3552 (111 tiles, per_core_M 2).
# The boundary is exactly 110/111 for every E > 1 tried (2, 4, 8, 16); E = 1, which is not the
# reuse path, is unaffected at T = 4096. Total output volume is not the bound: E=16 at 104 tiles
# is 159744 output tiles and passes, E=4 at 128 tiles is 49152 and hangs. Nor is it the
# requested core_grid: an explicit 8x8 still passes at 110 tiles and hangs at 111.
#
# The failure is a hang, not an exception, and it leaves the board needing tt-smi -r, so this is
# a limit to stay under rather than one to probe. B*S over 3520 is reachable in ordinary use,
# B=7 at S=512 being the smallest case.
MAX_TILE_ROWS_PER_CORE = 1

# 110 tiles is where the boundary was measured, and it did not move with the requested core grid:
# an explicit 8x8 passed at 110 and hung at 111, same as the full 11x10. So the limit is a
# property of the silicon here, not of the grid, and deriving it from the core count alone would
# raise it on a wider board rather than keep it. A 13x10 Blackhole, which this repo also targets,
# would derive 130 rows and send a 128-tile matmul down the single-pass branch: a hung board, not
# a failed assert. The grid still bounds it from below, since a narrower one cannot place 110
# rows one per core, so take the smaller of the two.
MAX_TILE_ROWS_MEASURED_SAFE = 110


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
        self.num_experts = config.num_experts

        grid = tt_config.core_grid
        tile_rows = min(grid.x * grid.y * MAX_TILE_ROWS_PER_CORE, MAX_TILE_ROWS_MEASURED_SAFE)
        self.max_tokens_per_pass = tile_rows * ttnn.TILE_SIZE

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

    def _weighted_expert_sum(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run every expert over one pass of tokens and sum the routed contributions.

        The bias is not added here: it is shared across experts and across passes, so it is
        added once to the assembled result rather than once per pass.

        Args:
            x: (1, 1, t, H) flat token activations, t at most max_tokens_per_pass.
            dense_weights: (1, 1, t, E) from TtNomicRouter, zero off the top-k.

        Returns:
            ttnn.Tensor: (1, 1, t, H).
        """
        tokens, hidden = x.shape[-2], x.shape[-1]

        hidden_states = ttnn.matmul(x, self.w1, compute_kernel_config=self.tt_config.compute_kernel_config)
        activated = ttnn.gelu(hidden_states)
        ttnn.deallocate(hidden_states)

        per_expert = ttnn.matmul(activated, self.w2, compute_kernel_config=self.tt_config.compute_kernel_config)
        ttnn.deallocate(activated)

        # (1, 1, t, E) -> (1, E, t, 1), whose trailing singleton broadcasts over the hidden axis.
        gate = ttnn.permute(dense_weights, (0, 3, 2, 1))
        gated = ttnn.multiply(per_expert, gate)
        ttnn.deallocate(per_expert)
        ttnn.deallocate(gate)

        summed = ttnn.experimental.fast_reduce_nc(
            gated, dims=[1], compute_kernel_config=self.tt_config.compute_kernel_config
        )
        ttnn.deallocate(gated)

        # fast_reduce_nc returns the tile-padded row count, not t: at t=74 it reports 96 rows
        # with the trailing 22 zero. The data is right, the logical shape is not.
        if summed.shape[-2] != tokens:
            trimmed = ttnn.slice(summed, [0, 0, 0, 0], [1, 1, tokens, hidden])
            ttnn.deallocate(summed)
            summed = trimmed
        return summed

    def forward(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run every expert, weight the outputs by the routing, and sum them.

        Tokens are processed in passes of at most max_tokens_per_pass, since beyond that the
        broadcast-batch matmul hangs. One pass covers B*S up to 3520 here; the chunking shapes in
        the bring-up tests exceed that and take the multi-pass branch, which is the point of them.

        Args:
            x: (1, 1, T, H) flat token activations.
            dense_weights: (1, 1, T, E) from TtNomicRouter, zero off the top-k.

        Returns:
            ttnn.Tensor: (1, 1, T, H).
        """
        tokens, hidden = x.shape[-2], x.shape[-1]

        if tokens <= self.max_tokens_per_pass:
            summed = self._weighted_expert_sum(x, dense_weights)
        else:
            passes = []
            for begin in range(0, tokens, self.max_tokens_per_pass):
                end = min(begin + self.max_tokens_per_pass, tokens)
                token_slice = ttnn.slice(x, [0, 0, begin, 0], [1, 1, end, hidden])
                weight_slice = ttnn.slice(dense_weights, [0, 0, begin, 0], [1, 1, end, self.num_experts])
                passes.append(self._weighted_expert_sum(token_slice, weight_slice))
                ttnn.deallocate(token_slice)
                ttnn.deallocate(weight_slice)

            summed = ttnn.concat(passes, dim=-2)
            for piece in passes:
                ttnn.deallocate(piece)

        out = ttnn.add(summed, self.bias)
        ttnn.deallocate(summed)
        return out
