# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``qsa_rows``: the QSA verify-form glue on the 32-row tile at R real rows, as fused programs.

Each program is bitwise against the op chain it replaces on the rows the tile carries. Rows past R are not zero on the
verify tile; every program masks or ignores them exactly as its chain does.
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from .. import qsa_block
from ..registry import BITWISE, FusedKernel, register

NAME = "qsa_rows"
PAGE = qsa_block.SCORE_CHUNK  # the all-gather page: 1024 bf16 block scores
DEVICES = qsa_block.DEVICES


def _score_rows_shape(score_rows, mask):
    shape, mshape = tuple(score_rows.shape), tuple(mask.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or not 1 <= shape[2] <= ttnn.TILE_SIZE
        or shape[3] % PAGE
        or score_rows.dtype != ttnn.bfloat16
        or score_rows.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(
            f"score rows must be ROW_MAJOR bf16 [1, 1, 1..32, k * {PAGE}], got {score_rows.layout} {shape}"
        )
    if mshape != shape or mask.dtype != ttnn.bfloat16 or mask.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"the block mask must be ROW_MAJOR bf16 of the rows' shape {shape}, got {mask.layout} {mshape}"
        )
    return shape[2], shape[3]


def score_blocks_rows(score_rows, mask, *, cluster_axis: int):
    """Program 1: the local block scores ``[1, 1, rows, blocks]`` bf16 ROW_MAJOR summed over the devices, then masked.

    The rows are all-gathered as 1024-column pages and merged by ``qsa_block.score_merge``; ``mask`` has the same
    shape and masks the rows past R as the chain does."""

    rows, blocks = _score_rows_shape(score_rows, mask)
    pages = blocks // PAGE
    # row r's chunk c at page r * pages + c; the gather stacks the devices' page runs, device-major
    paged = ttnn.reshape(score_rows, (1, 1, rows * pages, PAGE))
    gathered = ttnn.all_gather(paged, dim=2, cluster_axis=cluster_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(paged)
    if tuple(gathered.shape) != (1, 1, DEVICES * rows * pages, PAGE):
        raise RuntimeError(
            f"gathered score pages have shape {tuple(gathered.shape)}, expected [1, 1, {DEVICES * rows * pages}, {PAGE}]"
        )
    masked = qsa_block.score_merge(gathered, mask)
    ttnn.deallocate(gathered)
    return masked


def score_blocks_rows_composed(score_rows, mask, *, cluster_axis: int, topology=None):
    """The unfused op chain: ``ttnn.all_reduce`` over the devices, then the mask add."""

    _score_rows_shape(score_rows, mask)
    dram = ttnn.DRAM_MEMORY_CONFIG
    kwargs = {} if topology is None else {"topology": topology}
    scores = ttnn.all_reduce(score_rows, cluster_axis=cluster_axis, memory_config=dram, **kwargs)
    masked = ttnn.add(scores, mask, memory_config=dram, fast_and_approximate_mode=False)
    ttnn.deallocate(scores)
    return masked


def selection_rows(block_ids, sentinel_pad, block_offsets_rows, row_keep_bits, row_fill):
    """Program 3: the verify tile's sparse-attention token-id rows [1, 1, 32, 2080] uint32 from the top-k block ids.

    The decode ``qsa_selection_row`` program on the 32 rows, with a one-row ``sentinel_pad`` and per-row
    ``block_offsets_rows`` [1, 1, 32, 2048]."""

    shape = tuple(block_ids.shape)
    if shape != (1, 1, fp.TILE, qsa_block.BLOCK_IDS):
        raise ValueError(
            f"the verify selection takes the 32-row block ids [1, 1, 32, {qsa_block.BLOCK_IDS}], got {shape}"
        )
    return qsa_block.selection_row(block_ids, sentinel_pad, block_offsets_rows, row_keep_bits, row_fill)


def selection_rows_composed(block_ids, sentinel_pad_rows, block_offsets_rows, row_keep_bits, row_fill):
    """The unfused op chain (ttnn/qsa.py ``_materialize_rows_chunk``); ``sentinel_pad_rows`` is [1, 1, 32, 32]."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    starts = ttnn.bitwise_left_shift(block_ids, 2, memory_config=dram)
    repeated = ttnn.repeat_interleave(starts, repeats=qsa_block.COMPRESS_RATIO, dim=3, memory_config=dram)
    expanded = ttnn.add(repeated, block_offsets_rows, memory_config=dram)
    template = ttnn.concat([expanded, sentinel_pad_rows], dim=3, memory_config=dram)
    kept = ttnn.bitwise_and(template, row_keep_bits, memory_config=dram)
    out = ttnn.bitwise_or(kept, row_fill, memory_config=dram)
    for t in (starts, repeated, expanded, template, kept):
        ttnn.deallocate(t)
    return out


register(
    FusedKernel(
        name=NAME,
        replaces=(
            "the QSA verify-form glue on the 32-row tile: program 1 the indexer scores' all-reduce composite + mask add "
            "(6 programs/layer) as one all-gather + the fused score merge over the rows; program 2 the main tail with the "
            "rows' KV stage; program 3 the selection's integer chain (6 programs/layer) as the decode selection program"
        ),
        tolerance=BITWISE,
        fused=score_blocks_rows,
        composed=score_blocks_rows_composed,
        gate=None,  # the family's device tests are the component gate
    )
)

# program 2: the main tail with the verify rows' KV stage (the attribute is the function; the module stays importable
# through sys.modules)
from .main_tail_rows import kv_stage, main_tail_rows, main_tail_rows_composed  # noqa: E402

# program 5: the post-attention glue as one 48-core program; kernels, input contract and composed chain live in
# post_attention_rows.py, the launch here
from .post_attention_rows import (  # noqa: E402
    ITEMS as PA_ITEMS,
    PA_COMPUTE,
    PA_NAME,
    PA_READER,
    PA_WRITER,
    _check_inputs as _pa_inputs,
    post_attention_rows_composed,
)


def post_attention_rows_admits(attention, qg_ws, qg_first: int = 0) -> bool:
    """Program 5's input contract: the qg projection as one row tile ``[1, 1, 1..32, 3072]`` and sparse_sdpa's
    ROW_MAJOR ``[1, 32, rows, 256]``; any other form keeps the op chain."""

    if not (fp.is_row_tile(qg_ws) and fp.is_tile_width(qg_ws) and qg_ws.layout == ttnn.TILE_LAYOUT):
        return False  # the linear's tile shard, one row tile wide enough for the qg window
    rows = int(qg_ws.shape[-2])
    return (
        tuple(attention.shape) == (1, qsa_block.SPARSE_HEADS, rows, qsa_block.HEAD_DIM)
        and attention.dtype == ttnn.bfloat16
        and attention.layout == ttnn.ROW_MAJOR_LAYOUT
        and qg_first + qsa_block.QG_WIDTH // fp.TILE <= fp.tile_width_of(qg_ws) // fp.TILE
    )


def post_attention_rows(attention, qg_ws, *, memory_config=None, qg_first: int = 0):
    """sigmoid(gate) x attention for the six local heads -> the head-major ``[1, 1, rows, 1536]`` bf16 TILE row.

    ``attention`` is sparse_sdpa's ROW_MAJOR ``[1, 32, rows, 256]``; the gates are the second 256 columns of each head's
    512 in ``qg_ws`` from tile ``qg_first``. One core per (head, tile column)."""

    rows = _pa_inputs(attention, qg_ws, qg_first)
    mesh = qg_ws.device()
    out = fp.allocate(
        (1, 1, rows, qsa_block.OUT_WIDTH),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh,
        memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
    work = fp.split_work(len(PA_ITEMS), mesh)
    if any(w.count != 1 for w in work):
        raise RuntimeError(
            f"{PA_NAME} wants one core per output tile ({len(PA_ITEMS)}); the grid gives {len(work)} cores"
        )
    cores = fp.core_rectangle(work, mesh)
    cbs = [fp.cb_descriptor(cb, ttnn.bfloat16, fp.TILE_BYTES[ttnn.bfloat16], 1, cores) for cb in (0, 1, 2, 3, 16)]
    reader = fp.reader_kernel(
        PA_READER,
        cores,
        [*fp.accessor_args(attention), *fp.accessor_args(qg_ws)],
        [
            (w.core, [attention.buffer_address(), qg_ws.buffer_address(), rows, *PA_ITEMS[w.start], qg_first])
            for w in work
        ],
    )
    compute = fp.compute_kernel(PA_COMPUTE, cores, [], [(w.core, []) for w in work], fp32_dest=True)
    writer = fp.writer_kernel(
        PA_WRITER,
        cores,
        fp.accessor_args(out),
        [
            (w.core, [out.buffer_address(), PA_ITEMS[w.start][0] * qsa_block.HEAD_TILES + PA_ITEMS[w.start][1]])
            for w in work
        ],
    )
    # the output is head-sharded on dim 3
    meta = fp.program_meta(
        NAME,
        "post_attention_rows",
        outputs=((out, 3),),
    )
    return fp.run_program([attention, qg_ws, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


register(
    FusedKernel(
        name=PA_NAME,
        replaces=(
            "the QSA verify tile's post-attention glue (12 programs per attention layer: the gate's copy, slices and "
            "concat; the local heads' slice, tilize, sigmoid, multiply, slices, concat and the move into the "
            "out-projection shard) as one 48-core program, one core per (head, tile column)"
        ),
        tolerance=BITWISE,
        fused=post_attention_rows,
        composed=post_attention_rows_composed,
        admits=post_attention_rows_admits,
        gate=None,
    )
)


__all__ = [
    "NAME",
    "post_attention_rows_admits",
    "PA_NAME",
    "post_attention_rows",
    "post_attention_rows_composed",
    "kv_stage",
    "main_tail_rows",
    "main_tail_rows_composed",
    "score_blocks_rows",
    "score_blocks_rows_composed",
    "selection_rows",
    "selection_rows_composed",
]
