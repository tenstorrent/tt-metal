# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The expert FFN bank, the TTNN form of reference.NomicExperts.

Upstream gathers each expert's tokens by value, runs them, and scatter-adds the results back. This
runs that loop on the device, each token through its two chosen experts only. The router hands
over its top-2 selection, and each window of at most ROUTED_MAX_TOKENS tokens runs

    routing_metadata  the window's rows sorted by expert into 32-row aligned regions, R of them
      dispatch        (1, 1, R, H)  each slot's token row in its expert's region
      w1, gelu        (1, 1, R, F)  each 32-row block times its own expert's w1 (grouped_matmul)
      w2              (1, 1, R, H)  each block times its expert's w2, written row-major
      combine         (1, 1, t, H)  each token's two rows weighted and summed, the bias added once

R is 32 * (ceil(2t / 32) + E - 1), about a quarter of the E * t rows of running every token through
every expert, which the port did before: a stacked dense pass up to 320 tokens, a transposed one
above. The routed pass takes an 8x512 forward from 25.1 to 12.8 ms and, from 32 tokens up, cost
less host latency in 23 of 24 runs. On the device it cost 89 us a MoE layer more than the stacked
pass at 32 tokens; with per-expert lanes for small windows (PER_EXPERT_LANES_MAX_TOKENS_W1 and _W2)
it costs 22 us more at 32 tokens and 18 us less at 128. Its pooled-cosine tail on random ids is
within a few draws of theirs: 32 against 31 of 512 over 5e-3 at 1x128, 2x64, 2x37 and 2x512, 62
against 57 at 256 to 320 tokens.

The pass keeps the tokens on the rows, where a bfloat8_b tile's shared exponents never mix a
padding row with a real one, and gathers the real token rows only.
"""

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    activation_memory_config,
    pack_expert_weights,
    stack_expert_columns,
    to_device,
    transpose_linear_weight,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from models.experimental.nomic_embed_text_v2_moe.tt.router import Routing

# The largest window of a routed pass: routing_metadata takes 65536 slots, top_k x tokens. A larger
# window would also hold more of its intermediates in DRAM: at this size Xg, the w1 output and the
# expert rows take 0.43 GB.
ROUTED_MAX_TOKENS = 32768

# The largest windows whose w1 and whose w2 run grouped_matmul's per-expert lanes. A core row then
# takes one expert's blocks and reads that expert's weights itself, so the experts' weights load at
# once rather than one after another, but the busiest expert's blocks all fall on one row. Measured
# on layer 1 with its router's routing: w1 66.7 against 97.5 us at 32 tokens and 85.4 against 98.8
# at 256, but 105.2 against 101.0 at 512; w2 68.8 against 113.6 us at 32 and 99.0 against 115.4 at
# 512, 113.5 against 115.1 at 768. Each limit sits a step below its last win: w1 still won at 330
# tokens but lost 19 us there with half the tokens on one expert, and w2's 1.6 us at 768 is inside
# the noise.
PER_EXPERT_LANES_MAX_TOKENS_W1 = 256
PER_EXPERT_LANES_MAX_TOKENS_W2 = 512

# A routed pass's gathered rows, w1 output and expert rows go to L1 when each bank's share is at most
# this. The tightest window is the largest whose w1 output still fits, about 4300 tokens: grouped
# w2's circular buffers take 848 KB of the 1.46 MB a core has for them and L1 buffers together, its
# input (the w1 output) 256 KB and its output 124 KB, which leaves 226 KB; past it the w1 output goes
# to DRAM. So a pass frees the gathered rows (125 KB there) once w1 has read them, and a one-pass
# layer its dispatch table once dispatch has. Larger windows keep the gathered rows and the expert
# rows in L1 up to about 9200 tokens, 256 KB a bank each beside w1's 590 KB or w2's 848 KB.
ROUTED_L1_MAX_BANK_BYTES = 256 << 10
BFLOAT16_BYTES = 2


def routed_windows(tokens: int, max_tokens: int = ROUTED_MAX_TOKENS) -> list[tuple[int, int]]:
    """(token_start, num_tokens) of each routed pass: as few as max_tokens allows, of one size.

    routing_metadata takes windows that start on a tile row. One size, rather than full passes and
    a remainder, keeps every pass of a split batch above half of max_tokens: 70000 tokens run as
    passes of 23360, 23360 and 23280, not 32768, 32768 and 4464.
    """
    passes = ttnn.core.divup(tokens, max_tokens)
    size = ttnn.core.divup(ttnn.core.divup(tokens, passes), ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return [(start, min(size, tokens - start)) for start in range(0, tokens, size)]


class TtNomicExperts(LightweightModule):
    """All eight experts' weights, each expert's slab side by side, plus one bias shared across them.

    The bias is added once after the weighted sum. Adding it inside the per-expert loop scales
    it by the routed-weight sum, leaving an offset of (sum(w) - 1) * bias, which is real because
    the weights are not renormalized. That offset is nearly constant across tokens and PCC
    mean-centres, so the wrong version still scores 0.9999998; a per-token projection onto the
    bias sees it (test_shared_bias_is_added_after_the_weighted_sum).

    w2 is kept transposed per expert, (E, H, F), each expert's slab side by side on the device. That
    host form is also the shape of the one silent mistake: viewing the checkpoint's (E*F, H) block as
    (E, H, F) rather than (E, F, H) is an equally legal reshape, since E*F*H is symmetric in F and H.
    No inner dimension can catch it, so the module PCC tests are the guard, and
    test_misoriented_w2_decorrelates pins that they do through set_w2.

    Each step of a pass is a method of its own, so that the operator tests run the programs a pass
    runs.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.num_experts = config.num_experts
        self.max_tokens_per_routed_pass = ROUTED_MAX_TOKENS

        w1, w2 = state_dict[f"{state_dict_prefix}mlp.w1"], state_dict[f"{state_dict_prefix}mlp.w2"]
        _, w2 = pack_expert_weights(w1, w2, config)
        # The checkpoint's (E*F, H) block transposed on the host, every expert's (H, F) slab side by
        # side: a device transpose of a bfloat8_b tensor would regroup its shared exponents.
        self.w1_columns = to_device(
            transpose_linear_weight(w1).reshape(1, 1, config.hidden_size, -1),
            device,
            dtype=tt_config.expert_weight_dtype,
        )
        self.set_w2(w2.transpose(-2, -1).contiguous(), device)
        self.bias = to_device(
            state_dict[f"{state_dict_prefix}bias"].reshape(1, 1, 1, config.hidden_size),
            device,
            dtype=tt_config.weight_dtype,
        )

    def set_w2(self, w2_per_expert: torch.Tensor, device) -> None:
        """w2 on device from its (1, E, H, F) host form, every expert's (H, F) slab side by side."""
        self.w2_columns = to_device(
            stack_expert_columns(w2_per_expert), device, dtype=self.tt_config.expert_weight_dtype
        )

    def _windows(self, tokens: int) -> list[tuple[int, int]]:
        """The routed passes of a layer of `tokens` tokens; see routed_windows."""
        return routed_windows(tokens, self.max_tokens_per_routed_pass)

    def routed_memory_config(self, pages: int, page_bytes: int) -> ttnn.MemoryConfig:
        """L1 for a routed-pass intermediate of `pages` pages when its share of each bank fits
        ROUTED_L1_MAX_BANK_BYTES, DRAM otherwise."""
        bank_bytes = ttnn.core.divup(pages, self.tt_config.l1_banks) * page_bytes
        return ttnn.L1_MEMORY_CONFIG if bank_bytes <= ROUTED_L1_MAX_BANK_BYTES else ttnn.DRAM_MEMORY_CONFIG

    def dispatch_table(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, T, H) -> the row-major rows every routed pass of the layer gathers from.

        Untilized once a layer: ttnn.embedding untilizes a tile-layout table on every call. It lives
        through all the layer's passes, so it takes L1 only when the layer is one pass.
        """
        tokens, hidden = x.shape[-2], x.shape[-1]
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        if len(self._windows(tokens)) == 1:
            memory_config = self.routed_memory_config(tokens, hidden * BFLOAT16_BYTES)
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)

    def dispatch(self, x_rows: ttnn.Tensor, gather_index: ttnn.Tensor) -> ttnn.Tensor:
        """routing_metadata's (1, R) gather_index -> (1, 1, R, H), row r a copy of row gather_index[r] of x_rows.

        The ids are global, so every pass of a layer gathers from the one dispatch_table. The copy
        is exact: ttnn.embedding moves whole rows and tilizes them without arithmetic.
        """
        rows, hidden = gather_index.shape[-1], x_rows.shape[-1]
        tiles = rows // ttnn.TILE_SIZE * hidden // ttnn.TILE_SIZE
        dispatched = ttnn.embedding(
            gather_index,
            x_rows,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.routed_memory_config(tiles, ttnn.tile_size(ttnn.bfloat16)),
        )
        return ttnn.reshape(dispatched, (1, 1, dispatched.shape[-2], dispatched.shape[-1]))

    def grouped_w1(
        self, dispatched: ttnn.Tensor, expert_counts: ttnn.Tensor, per_expert_lanes: bool = False
    ) -> ttnn.Tensor:
        """dispatch's (1, 1, R, H) rows -> (1, 1, R, F), each 32-row block through its expert's w1 and GELU.

        The expert of each block follows from expert_counts on the device. Blocks past the last
        expert's region are neither computed nor written; nothing reads them. per_expert_lanes picks
        grouped_matmul's schedule, which leaves the output bit for bit the same.
        """
        dtype = self.tt_config.expert_intermediate_dtype
        tiles = dispatched.shape[-2] // ttnn.TILE_SIZE * self.w1_columns.shape[-1] // self.num_experts // ttnn.TILE_SIZE
        return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
            dispatched,
            self.w1_columns,
            expert_counts,
            dtype=dtype,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, 1.0 if self.tt_config.routed_fast_gelu else 0.0),
            memory_config=self.routed_memory_config(tiles, ttnn.tile_size(dtype)),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.ROUTED_W1),
            per_expert_lanes=per_expert_lanes,
        )

    def grouped_w2(
        self, activated: ttnn.Tensor, expert_counts: ttnn.Tensor, per_expert_lanes: bool = False
    ) -> ttnn.Tensor:
        """grouped_w1's (1, 1, R, F) -> (1, 1, R, H) bfloat16 rows, each block through its expert's w2.

        Row-major, so combine reads each token's rows whole. w2_columns is read transposed.
        per_expert_lanes as in grouped_w1.
        """
        rows, hidden = activated.shape[-2], self.w2_columns.shape[-2]
        return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
            activated,
            self.w2_columns,
            expert_counts,
            dtype=ttnn.bfloat16,
            transpose_weight=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.routed_memory_config(rows, hidden * BFLOAT16_BYTES),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.ROUTED_W2),
            per_expert_lanes=per_expert_lanes,
        )

    def combine(
        self, expert_rows: ttnn.Tensor, combine_index: ttnn.Tensor, combine_weights: ttnn.Tensor, tokens: int
    ) -> ttnn.Tensor:
        """grouped_w2's (1, 1, R, H) rows -> (1, 1, t, H): each token's top-k rows weighted and summed, plus the bias.

        A padding token reads row 0, always a real slot's row, at weight 0, so the tile padding of
        the result holds the bias: it stays finite into the next attention, which non-finite padded
        keys corrupt.
        """
        return ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
            expert_rows,
            combine_index,
            combine_weights,
            self.bias,
            num_tokens=tokens,
            # The module's output, placed as the blocks place their activations.
            memory_config=activation_memory_config((1, 1, tokens, expert_rows.shape[-1])),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.REDUCE),
        )

    def forward(self, x: ttnn.Tensor, routing: Routing) -> ttnn.Tensor:
        """(1, 1, T, H) -> (1, 1, T, H): each token through its top-k experts only, the bias included.

        Each intermediate is freed as soon as its consumer has run, which the L1 budget of
        ROUTED_L1_MAX_BANK_BYTES relies on. The windows' results are joined with ttnn.concat.

        Args:
            x: (1, 1, T, H) flat token activations, bfloat16.
            routing: TtNomicRouter's selection; it stays the caller's to free.

        Returns:
            ttnn.Tensor: (1, 1, T, H).
        """
        if x.dtype != ttnn.bfloat16:
            raise ValueError(f"the routed pass takes bfloat16 activations, got {x.dtype}")
        windows = self._windows(x.shape[-2])
        x_rows = self.dispatch_table(x)
        results = []
        for window, (token_start, num_tokens) in enumerate(windows):
            (
                gather_index,
                combine_index,
                combine_weights,
                expert_counts,
            ) = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                routing.indices,
                routing.values,
                num_experts=self.num_experts,
                token_start=token_start,
                num_tokens=num_tokens,
            )
            dispatched = self.dispatch(x_rows, gather_index)
            ttnn.deallocate(gather_index)
            if window == len(windows) - 1:
                ttnn.deallocate(x_rows)
            activated = self.grouped_w1(
                dispatched, expert_counts, per_expert_lanes=num_tokens <= PER_EXPERT_LANES_MAX_TOKENS_W1
            )
            ttnn.deallocate(dispatched)
            expert_rows = self.grouped_w2(
                activated, expert_counts, per_expert_lanes=num_tokens <= PER_EXPERT_LANES_MAX_TOKENS_W2
            )
            ttnn.deallocate(activated)
            results.append(self.combine(expert_rows, combine_index, combine_weights, num_tokens))
            for tensor in (combine_index, combine_weights, expert_counts, expert_rows):
                ttnn.deallocate(tensor)
        if len(results) == 1:
            return results[0]
        out = ttnn.concat(results, dim=-2)
        for piece in results:
            ttnn.deallocate(piece)
        return out
