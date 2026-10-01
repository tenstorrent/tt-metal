# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""qsa_rows program 5 (``qsa_rows_post_attention``): the verify tile's post-attention glue as one 48-core program
(the kernels, the input contract and the composed chain; the launch and the registry entry sit in the family's
``__init__``, where the program-meta static reads them).

After ``sparse_sdpa`` the chain runs twelve programs per attention layer on the 32-row tile: the gate's copy of the qg
projection to DRAM, six head slices and their concat (``_main_tail_rows_step``), the slice of the local heads out of the
sparse output, its tilize, the sigmoid, the multiply, six head slices, their concat into the head-major
``[1, 1, 32, 1536]`` row and its move into the out-projection's activation shard (``_sparse_value_attention_rows`` /
``_project_output_rows``).  The decode's ``qsa_post_attention`` program does the same for one row on six cores (one per
head, a serial row placement), and on the 32-row tile it measured slower than the chain (0.087 against 0.059 ms per
call, 2026-09-26).  This form puts one core on every (head, tile column) of the output -- 48 cores, one tile each: the
reader takes the head's gate tile straight from the qg shard and places the head's rows' 32-column window into one
tile (the chain's slice + tilize, zero rows past the tile's), the compute runs the decode kernel's two ops on that tile
(the SFPU sigmoid packed bf16, then the SFPU product with the bf16-dest rounding: the chain's unary sigmoid and
binary_ng multiply), and the writer lands the tile in the activation shard.  Every element sees the chain's operations
in the chain's order, so the program is BITWISE against the composed chain; the family's device test holds it on the
1x4 line.  Opt-in (``QWEN38_FUSED=qsa_rows_post_attention``) until its pass pair."""

from __future__ import annotations

import ttnn

from .. import program as fp
from .. import qsa_block

NAME = "qsa_rows"
PA_NAME = "qsa_rows_post_attention"
BF16 = ttnn.bfloat16
HEAD_DIM, HEAD_TILES, LOCAL_HEADS, SPARSE_HEADS = (
    qsa_block.HEAD_DIM,
    qsa_block.HEAD_TILES,
    qsa_block.LOCAL_HEADS,
    qsa_block.SPARSE_HEADS,
)
OUT_WIDTH, QG_WIDTH = qsa_block.OUT_WIDTH, qsa_block.QG_WIDTH
TILE_BF16 = fp.TILE_BYTES[BF16]
ITEMS = [(head, column) for head in range(LOCAL_HEADS) for column in range(HEAD_TILES)]  # 48 output tiles
PA_READER = fp.kernel_source(NAME, "pa_rows_reader.cpp")
PA_COMPUTE = fp.kernel_source(NAME, "pa_rows_compute.cpp")
PA_WRITER = fp.kernel_source(NAME, "pa_rows_writer.cpp")


def _check_inputs(attention, qg_ws, qg_first: int) -> int:
    rows = fp.rows_of(qg_ws)
    qsa_block._window(qg_ws, qg_first, QG_WIDTH, "qg")
    if (
        tuple(attention.shape) != (1, SPARSE_HEADS, rows, HEAD_DIM)
        or attention.dtype != BF16
        or attention.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(
            f"attention must be ROW_MAJOR bf16 [1, {SPARSE_HEADS}, {rows}, {HEAD_DIM}], got "
            f"{attention.layout} {attention.dtype} {tuple(attention.shape)}"
        )
    return rows


def post_attention_rows_composed(attention, qg_ws, *, memory_config=None):
    """The chain on the tile, op for op as ttnn/qsa.py runs it (``_main_tail_rows_step``'s gate,
    ``_sparse_value_attention_rows`` after sparse_sdpa, ``_project_output_rows``'s activation move)."""

    rows = _check_inputs(attention, qg_ws, 0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    qg = ttnn.to_memory_config(qg_ws, dram)
    gate_heads = [
        ttnn.slice(
            qg,
            (0, 0, 0, head * 2 * HEAD_DIM + HEAD_DIM),
            (1, 1, rows, (head + 1) * 2 * HEAD_DIM),
            memory_config=dram,
        )
        for head in range(LOCAL_HEADS)
    ]
    gate = ttnn.concat(gate_heads, dim=1, memory_config=dram)
    local = ttnn.slice(attention, (0, 0, 0, 0), (1, LOCAL_HEADS, rows, HEAD_DIM), memory_config=dram)
    local_tiled = ttnn.to_layout(local, ttnn.TILE_LAYOUT, memory_config=dram)
    activated_gate = ttnn.sigmoid(gate, memory_config=dram)
    gated = ttnn.mul(local_tiled, activated_gate, memory_config=dram)
    heads = [
        ttnn.slice(gated, (0, head, 0, 0), (1, head + 1, rows, HEAD_DIM), memory_config=dram)
        for head in range(LOCAL_HEADS)
    ]
    local_flat = ttnn.concat(heads, dim=3, memory_config=dram)
    out = ttnn.to_memory_config(local_flat, memory_config or dram)
    for t in (qg, *gate_heads, gate, local, local_tiled, activated_gate, gated, *heads):
        ttnn.deallocate(t)
    if out.buffer_address() != local_flat.buffer_address():
        ttnn.deallocate(local_flat)
    return out


__all__ = ["PA_NAME", "ITEMS", "PA_READER", "PA_COMPUTE", "PA_WRITER", "_check_inputs", "post_attention_rows_composed"]
