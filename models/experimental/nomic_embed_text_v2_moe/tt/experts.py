# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The expert FFN bank, the TTNN form of reference.NomicExperts.

Upstream gathers each expert's tokens by value, runs them, and scatter-adds the results back. This
runs that loop on the device, each token through its two chosen experts only. The router hands
over its scores, and each window of at most ROUTED_MAX_TOKENS tokens runs

    routing_metadata  each token's top-2 of the scores, and the window's rows sorted by expert
                      into 32-row aligned regions, R of them
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

from typing import Sequence

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
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import grouped_matmul_footprint, l1_depth, l1_floor
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

# The grouped matmuls read w1 and w2 as chunked weights, contiguous runs of this many bfloat8_b tiles in one DRAM bank
# each. w1's slice a core is 216 tiles: of 12 to 216, 36 (38 KB) was best, 8x512 162.6 -> 160.9 us; 2x512 is flat
# from 36 to 60 and slower below.
EXPERT_W1_CHUNK_TILES = 36
# w2's chunks: grouped_matmul_w2 splits each share's 288-tile slice into two K halves of 144, so a chunk is 48 tiles.
# grouped_matmul_w1's w2 path (windows grouped_matmul_w2 does not take) reads the same weight bit for bit.
EXPERT_W2_CHUNK_TILES = 48
# grouped_matmul_w2's core map is built for this compute grid; elsewhere w2 runs grouped_matmul_w1's path.
GROUPED_MATMUL_W2_GRID = (11, 10)

# A routed pass's gathered rows, w1 output and expert rows go to L1 when each bank's share is at most
# this. The tightest window is the largest whose w1 output still fits, about 4300 tokens: grouped
# w2's circular buffers take 848 KB of the 1.46 MB a core has for them and L1 buffers together, its
# input (the w1 output) 256 KB and its output 124 KB, which leaves 226 KB; past it the w1 output goes
# to DRAM. So a pass frees the gathered rows (125 KB there) once w1 has read them, and a one-pass
# layer its dispatch table once dispatch has. Larger windows keep the gathered rows and the expert
# rows in L1 up to about 9200 tokens, 256 KB a bank each beside w1's 590 KB or w2's 848 KB.
ROUTED_L1_MAX_BANK_BYTES = 256 << 10
BFLOAT16_BYTES = 2

# The gather's circular buffers per core: two slots of a unit's row tiles and of its output tiles, a
# unit at most 8 tiles wide. A routed intermediate stays in L1 only if the op that runs beside it still
# has room for its buffers (this, or grouped_w1's or grouped_w2's, matmul_config.grouped_matmul_footprint)
# below it and the live tensors, with _ROUTED_L1_MARGIN to spare.
_GATHER_CB_BYTES = 64 << 10
_ROUTED_L1_MARGIN = 16 << 10

# The most spare tile rows a gather writes (dispatch): every token count up to 8192 whose expert rows
# outgrow the gathered rows' slot gets it back with one to three.
MAX_SPARE_TILE_ROWS = 3


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
        self.device = device
        self.num_experts = config.num_experts
        self.hidden, self.intermediate = config.hidden_size, config.intermediate_size
        self.top_k = config.moe_top_k
        self.max_tokens_per_routed_pass = ROUTED_MAX_TOKENS

        w1, w2 = state_dict[f"{state_dict_prefix}mlp.w1"], state_dict[f"{state_dict_prefix}mlp.w2"]
        _, w2 = pack_expert_weights(w1, w2, config)
        # The checkpoint's (E*F, H) block transposed on the host, every expert's (H, F) slab side by
        # side: a device transpose of a bfloat8_b tensor would regroup its shared exponents.
        self.w1 = self._chunked(
            transpose_linear_weight(w1).reshape(1, 1, config.hidden_size, -1), device, False, EXPERT_W1_CHUNK_TILES
        )
        self.w1_options = dict(weight_chunk_tiles=EXPERT_W1_CHUNK_TILES, weight_n=self.intermediate)
        self.set_w2(w2.transpose(-2, -1).contiguous(), device)
        self.bias = to_device(
            state_dict[f"{state_dict_prefix}bias"].reshape(1, 1, 1, config.hidden_size),
            device,
            dtype=tt_config.weight_dtype,
        )
        # The token ids of each spare tile row a gather may write past routing_metadata's (dispatch).
        self.spare_row_ids = ttnn.from_torch(
            torch.arange(ttnn.TILE_SIZE, dtype=torch.int32).reshape(1, ttnn.TILE_SIZE),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def set_w2(self, w2_per_expert: torch.Tensor, device) -> None:
        """w2 on device from its (1, E, H, F) host form, every expert's (H, F) slab side by side, read transposed."""
        self.w2 = self._chunked(stack_expert_columns(w2_per_expert), device, True, EXPERT_W2_CHUNK_TILES)
        self.w2_options = dict(weight_chunk_tiles=EXPERT_W2_CHUNK_TILES, weight_n=self.hidden)

    def _chunked(self, columns: torch.Tensor, device, transposed: bool, chunk_tiles: int) -> ttnn.Tensor:
        """An expert projection's (1, 1, R, E * C) weight as grouped_matmul's chunked weight: the bfloat8_b tiles
        quantized on the host as a TILE weight, then copied whole into the chunk layout."""
        tiles = ttnn.from_torch(columns, dtype=self.tt_config.expert_weight_dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.experimental.nomic_embed_text_v2_moe.prepare_grouped_matmul_w1_weight(
            tiles, self.num_experts, transpose_weight=transposed, chunk_tiles=chunk_tiles, device=device
        )

    def _windows(self, tokens: int) -> list[tuple[int, int]]:
        """The routed passes of a layer of `tokens` tokens; see routed_windows."""
        return routed_windows(tokens, self.max_tokens_per_routed_pass)

    def bank_bytes(self, pages: int, page_bytes: int) -> int:
        return ttnn.core.divup(pages, self.tt_config.l1_banks) * page_bytes

    @staticmethod
    def _tile_pages(tensor: ttnn.Tensor) -> tuple[int, int]:
        """A tile-layout tensor's pages and page bytes."""
        tiles = tensor.shape[-2] // ttnn.TILE_SIZE * tensor.shape[-1] // ttnn.TILE_SIZE
        return tiles, ttnn.tile_size(tensor.dtype)

    def routed_memory_config(
        self,
        pages: int,
        page_bytes: int,
        live: Sequence[ttnn.Tensor],
        cb_bytes: int,
        freed: tuple[int, int] | None = None,
    ) -> ttnn.MemoryConfig:
        """L1 for a routed-pass intermediate of `pages` pages when its share of each bank fits
        ROUTED_L1_MAX_BANK_BYTES and the op that runs beside it still has cb_bytes for its buffers
        below it and the live tensors; DRAM otherwise.

        L1 tensors are allocated from the top down, each in a high gap it fits: in freed, the depth
        (l1_depth) and bank bytes of a slot just freed, when it fits there, and at worst right below
        the deepest live tensor otherwise. Each op below adds its own inputs to the `live` it is given.
        """
        fits = self._fits_l1(self.bank_bytes(pages, page_bytes), l1_floor(live, self.tt_config), cb_bytes, freed)
        return ttnn.L1_MEMORY_CONFIG if fits else ttnn.DRAM_MEMORY_CONFIG

    def _fits_l1(self, bank_bytes: int, deepest: int, cb_bytes: int, freed: tuple[int, int] | None = None) -> bool:
        """routed_memory_config's test, with the live tensors reaching `deepest` bytes a bank from the top."""
        if freed is not None and bank_bytes <= freed[1]:
            bottom = max(deepest, freed[0])
        else:
            bottom = deepest + bank_bytes
        fits = bank_bytes <= ROUTED_L1_MAX_BANK_BYTES
        return fits and bottom + cb_bytes + _ROUTED_L1_MARGIN <= self.tt_config.l1_cb_bytes

    def _spare_tile_rows(self, rows: int, hidden: int, deepest: int) -> int:
        """The tile rows the gather writes past routing_metadata's `rows` so that w2's output stays in L1, if any.

        w2's output takes the slot the gathered rows free (grouped_w2's freed), but its 1536-byte row pages
        can need up to 2 KB more of a bank than the gathered rows' 2048-byte tiles. At 3744, 3904, 3968,
        4128 and 4192 tokens neither that slot nor the room below the w1 output, where w2's buffers go,
        held it, and it went to DRAM. Spare tile rows widen the slot, but a row's 24 tiles spread over the
        banks, so it can take two or three to add a tile to the fullest one; grouped_matmul neither
        computes nor writes blocks past the last expert's, and the combine never reads them. The
        placements are predicted as routed_memory_config makes them, each at worst right below the
        deepest tensor.
        """
        dtype = self.tt_config.expert_intermediate_dtype
        gather_cb = self.footprint(self.hidden, self.intermediate, ttnn.bfloat16, dtype)
        w2_cb = self.footprint(self.intermediate, self.hidden, dtype, ttnn.bfloat16)
        columns = self.intermediate
        tile = ttnn.TILE_SIZE

        def expert_rows_in_l1(count: int) -> bool:
            gathered = self.bank_bytes(count // tile * hidden // tile, ttnn.tile_size(ttnn.bfloat16))
            freed = (deepest + gathered, gathered) if self._fits_l1(gathered, deepest, gather_cb) else None
            below = freed[0] if freed else deepest
            w1_out = self.bank_bytes(count // tile * columns // tile, ttnn.tile_size(dtype))
            if self._fits_l1(w1_out, below, max(gather_cb, w2_cb)):
                below += w1_out
            return self._fits_l1(self.bank_bytes(count, hidden * BFLOAT16_BYTES), below, w2_cb, freed)

        if expert_rows_in_l1(rows):
            return 0
        return next((spare for spare in range(1, MAX_SPARE_TILE_ROWS + 1) if expert_rows_in_l1(rows + spare * tile)), 0)

    def footprint(self, k: int, n: int, input_dtype: ttnn.DataType, output_dtype: ttnn.DataType) -> int:
        """grouped_matmul's circular-buffer bytes a core for one of the experts' projections, (K, N) an expert's."""
        return grouped_matmul_footprint(
            k // ttnn.TILE_SIZE,
            n // ttnn.TILE_SIZE,
            self.tt_config.core_grid,
            input_dtype,
            self.tt_config.expert_weight_dtype,
            output_dtype,
            self.num_experts,
        )

    def dispatch_table_config(self, x: ttnn.Tensor, live: Sequence[ttnn.Tensor] = ()) -> ttnn.MemoryConfig:
        """Where dispatch_table puts x's rows. They live through all the layer's passes, so they take L1
        only when the layer is one pass."""
        tokens, hidden = x.shape[0] * x.shape[-2], x.shape[-1]
        if len(self._windows(tokens)) > 1:
            return ttnn.DRAM_MEMORY_CONFIG
        return self.routed_memory_config(tokens, hidden * BFLOAT16_BYTES, (*live, x), _GATHER_CB_BYTES)

    def dispatch_table(self, x: ttnn.Tensor, live: Sequence[ttnn.Tensor] = ()) -> ttnn.Tensor:
        """(1, 1, T, H) -> the row-major rows every routed pass of the layer gathers from, untilized once a layer:
        ttnn.embedding untilizes a tile-layout table on every call. The MoE layer has the router write them
        instead (TtNomicRouter.forward); this serves a call without them."""
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dispatch_table_config(x, live))

    def dispatch(self, x_rows: ttnn.Tensor, gather_index: ttnn.Tensor, live: Sequence[ttnn.Tensor] = ()) -> ttnn.Tensor:
        """routing_metadata's (1, R) gather_index -> (1, 1, R, H), row r a copy of row gather_index[r] of x_rows.

        The ids are global, so every pass of a layer gathers from the one dispatch_table. The gather
        is routed_combine without weights or bias: it tilizes the rows without arithmetic, bit for bit
        what ttnn.embedding returns, but splits each 32-row block's columns over the grid where
        ttnn.embedding gives a core whole blocks, 15 cores for the 480 rows of 128 tokens.

        Where w2's output stays in L1 only with them (_spare_tile_rows), the gather writes spare tile
        rows, each a copy of tokens 0 to 31, and R grows by as many.
        """
        rows, hidden = gather_index.shape[-1], x_rows.shape[-1]
        live = (*live, x_rows, gather_index)
        index = gather_index
        spare = self._spare_tile_rows(rows, hidden, l1_floor(live, self.tt_config))
        if spare:
            # One op, 1.7 us at 8480 rows; ttnn.pad takes 21.
            index = ttnn.concat(
                [gather_index, *[self.spare_row_ids] * spare], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            rows += spare * ttnn.TILE_SIZE
        tiles = rows // ttnn.TILE_SIZE * hidden // ttnn.TILE_SIZE
        dispatched = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
            x_rows,
            index,
            num_tokens=rows,
            # grouped_w1 runs beside it; its output plans for itself.
            memory_config=self.routed_memory_config(
                tiles,
                ttnn.tile_size(ttnn.bfloat16),
                live,
                self.footprint(self.hidden, self.intermediate, ttnn.bfloat16, self.tt_config.expert_intermediate_dtype),
            ),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.REDUCE),
        )
        if index is not gather_index:
            ttnn.deallocate(index)
        return dispatched

    def grouped_w1(
        self,
        dispatched: ttnn.Tensor,
        expert_counts: ttnn.Tensor,
        per_expert_lanes: bool = False,
        live: Sequence[ttnn.Tensor] = (),
        for_grouped_matmul_w2: bool = False,
    ) -> ttnn.Tensor:
        """dispatch's (1, 1, R, H) rows -> (1, 1, R, F), each 32-row block through its expert's w1 and GELU.

        The expert of each block follows from expert_counts on the device. Blocks past the last
        expert's region are neither computed nor written; nothing reads them. per_expert_lanes picks
        grouped_matmul's schedule, which leaves the output bit for bit the same.

        With for_grouped_matmul_w2, the output goes to grouped_matmul_w2's lane-local layout where it fits L1
        (_w2_input_config), which makes grouped_w2 run that op; otherwise as routed_memory_config places it.
        """
        dtype = self.tt_config.expert_intermediate_dtype
        tiles = dispatched.shape[-2] // ttnn.TILE_SIZE * self.intermediate // ttnn.TILE_SIZE
        live = (*live, dispatched, expert_counts)
        w1_cb = self.footprint(self.hidden, self.intermediate, dispatched.dtype, dtype)
        memory_config = None
        if for_grouped_matmul_w2:
            memory_config = self._w2_input_config(dispatched.shape[-2], live, w1_cb)
        if memory_config is None:
            # w1 writes it, w2 reads it: room for the larger of their buffers.
            memory_config = self.routed_memory_config(
                tiles,
                ttnn.tile_size(dtype),
                live,
                max(w1_cb, self.footprint(self.intermediate, self.hidden, dtype, ttnn.bfloat16)),
            )
        return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1(
            dispatched,
            self.w1,
            expert_counts,
            dtype=dtype,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, float(self.tt_config.routed_gelu_form)),
            memory_config=memory_config,
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.ROUTED_W1),
            per_expert_lanes=per_expert_lanes,
            **self.w1_options,
        )

    def _w2_cb_bytes(self) -> int:
        """grouped_matmul_w2's circular-buffer bytes a core, from the op."""
        return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2_circular_buffer_bytes(
            self.intermediate, self.hidden, 2, 2, self.num_experts, device=self.device
        )

    def _w2_input_config(self, rows: int, live: Sequence[ttnn.Tensor], w1_cb: int) -> ttnn.MemoryConfig | None:
        """grouped_matmul_w2's input layout for `rows` rows when its share of a core, beside the live tensors, still
        leaves both grouped matmuls their circular buffers (with _ROUTED_L1_MARGIN); None otherwise.

        The layout puts each block's quarter pieces on its own lane's cores, a shard a quarter, round-robin over the
        cores of its grid: a core holds ceil(shards / cores) of them.
        """
        grid = self.tt_config.core_grid
        if (grid.x, grid.y) != GROUPED_MATMUL_W2_GRID:
            return None
        config = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2_input_memory_config(
            rows, self.intermediate, device=self.device
        )
        spec = config.nd_shard_spec
        shard_rows, shard_columns = spec.shard_shape[0], spec.shard_shape[1]
        shards = rows // shard_rows * (self.intermediate // shard_columns)
        shard_bytes = (
            shard_rows
            // ttnn.TILE_SIZE
            * shard_columns
            // ttnn.TILE_SIZE
            * ttnn.tile_size(self.tt_config.expert_intermediate_dtype)
        )
        core_bytes = ttnn.core.divup(shards, spec.num_cores()) * shard_bytes
        # Not held to ROUTED_L1_MAX_BANK_BYTES: at 8x512 the pieces take 287 KB a core, and only this op reads them;
        # what has to fit is the buffers of both grouped matmuls below them.
        bottom = l1_floor(live, self.tt_config) + core_bytes
        cb_bytes = max(w1_cb, self._w2_cb_bytes())
        return config if bottom + cb_bytes + _ROUTED_L1_MARGIN <= self.tt_config.l1_cb_bytes else None

    def grouped_w2(
        self,
        activated: ttnn.Tensor,
        expert_counts: ttnn.Tensor,
        per_expert_lanes: bool = False,
        live: Sequence[ttnn.Tensor] = (),
        freed: tuple[int, int] | None = None,
    ) -> ttnn.Tensor:
        """grouped_w1's (1, 1, R, F) -> (1, 1, R, H) bfloat16 rows, each block through its expert's w2.

        Row-major, so combine reads each token's rows whole. w2 is read transposed.
        grouped_matmul_w2 runs it when grouped_w1 wrote its input in that op's layout; grouped_matmul_w1's
        transposed path otherwise, the two bit for bit the same.
        per_expert_lanes as in grouped_w1; freed as in routed_memory_config, the gathered rows' slot.
        """
        rows, hidden = activated.shape[-2], self.hidden
        live = (*live, activated, expert_counts)
        if activated.memory_config().nd_shard_spec is not None:
            return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2(
                activated,
                self.w2,
                expert_counts,
                memory_config=self.routed_memory_config(
                    rows, hidden * BFLOAT16_BYTES, live, self._w2_cb_bytes(), freed
                ),
                compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.ROUTED_W2),
                **self.w2_options,
            )
        return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1(
            activated,
            self.w2,
            expert_counts,
            dtype=ttnn.bfloat16,
            transpose_weight=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.routed_memory_config(
                rows,
                hidden * BFLOAT16_BYTES,
                live,
                self.footprint(self.intermediate, self.hidden, activated.dtype, ttnn.bfloat16),
                freed,
            ),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.ROUTED_W2),
            per_expert_lanes=per_expert_lanes,
            **self.w2_options,
        )

    def combine(
        self,
        expert_rows: ttnn.Tensor,
        combine_index: ttnn.Tensor,
        combine_weights: ttnn.Tensor,
        tokens: int,
        memory_config: ttnn.MemoryConfig | None = None,
        batch: int = 1,
    ) -> ttnn.Tensor:
        """grouped_w2's (1, 1, R, H) rows -> (1, 1, t, H): each token's top-k rows weighted and summed, plus the bias.

        batch > 1 writes (batch, 1, t / batch, H), page for page the same (each sequence whole tiles).

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
            # The module's output, placed as the blocks place their activations unless the caller says.
            memory_config=memory_config or activation_memory_config((1, 1, tokens, expert_rows.shape[-1])),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.REDUCE),
            batch=batch,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        routing: Routing,
        memory_config: ttnn.MemoryConfig | None = None,
        x_rows: ttnn.Tensor | None = None,
        batch: int = 1,
    ) -> ttnn.Tensor:
        """(1, 1, T, H) -> (1, 1, T, H): each token through its top-k experts only, the bias included.

        x may also be (B, 1, S, H) with S on the tile grid (T = B * S), read as the flat token axis; with batch = B
        a one-pass layer then writes (B, 1, S, H).

        Each intermediate is freed as soon as its consumer has run, which the L1 budget of
        ROUTED_L1_MAX_BANK_BYTES relies on, and is placed knowing which tensors are live beside it
        (routed_memory_config). The windows' results are joined with ttnn.concat.

        Args:
            x: (1, 1, T, H) flat token activations, bfloat16; in L1 the block's residual too.
            routing: TtNomicRouter's scores, from which routing_metadata selects; it stays the caller's to free.
            memory_config: where a one-pass layer writes its output; activation_memory_config by default.
            x_rows: x's dispatch_table when the caller has built it, which the forward frees.

        Returns:
            ttnn.Tensor: (1, 1, T, H).
        """
        if x.dtype != ttnn.bfloat16:
            raise ValueError(f"the routed pass takes bfloat16 activations, got {x.dtype}")
        windows = self._windows(x.shape[0] * x.shape[-2])
        if x_rows is None:
            x_rows = self.dispatch_table(x, routing)
        results = []
        for window, (token_start, num_tokens) in enumerate(windows):
            (
                gather_index,
                combine_index,
                combine_weights,
                expert_counts,
            ) = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                scores=routing.scores,
                top_k=self.top_k,
                num_experts=self.num_experts,
                token_start=token_start,
                num_tokens=num_tokens,
            )
            last = window == len(windows) - 1
            live = (x, *routing, *results, combine_index, combine_weights)
            dispatched = self.dispatch(x_rows, gather_index, (*live, expert_counts))
            ttnn.deallocate(gather_index)
            if last:
                ttnn.deallocate(x_rows)
            else:
                live += (x_rows,)
            activated = self.grouped_w1(
                dispatched,
                expert_counts,
                per_expert_lanes=num_tokens <= PER_EXPERT_LANES_MAX_TOKENS_W1,
                live=live,
                for_grouped_matmul_w2=num_tokens > PER_EXPERT_LANES_MAX_TOKENS_W2,
            )
            # The expert rows are the gathered rows' size, so they take that slot once it is freed.
            freed = None
            if dispatched.memory_config().buffer_type == ttnn.BufferType.L1:
                freed = (l1_depth(dispatched, self.tt_config), self.bank_bytes(*self._tile_pages(dispatched)))
            ttnn.deallocate(dispatched)
            expert_rows = self.grouped_w2(
                activated,
                expert_counts,
                per_expert_lanes=num_tokens <= PER_EXPERT_LANES_MAX_TOKENS_W2,
                live=live,
                freed=freed,
            )
            ttnn.deallocate(activated)
            results.append(
                self.combine(
                    expert_rows,
                    combine_index,
                    combine_weights,
                    num_tokens,
                    memory_config if len(windows) == 1 else None,
                    batch if len(windows) == 1 else 1,
                )
            )
            for tensor in (combine_index, combine_weights, expert_counts, expert_rows):
                ttnn.deallocate(tensor)
        if len(results) == 1:
            return results[0]
        out = ttnn.concat(results, dim=-2)
        for piece in results:
            ttnn.deallocate(piece)
        return out
