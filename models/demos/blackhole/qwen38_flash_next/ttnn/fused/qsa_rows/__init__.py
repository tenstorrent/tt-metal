# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``qsa_rows``: the QSA verify-form glue on the 32-row tile at R real rows, as fused programs.

``Qwen38TTNNQSA.forward_verify_generic`` runs the R rows P .. P + R - 1 of one pass on one 32-row tile (R = k + 1 the
MTP verify rows of the twelve backbone layers, R = 1 the MTP layer's draft row padded in place); its glue around the
kept kernels (the all-gather of the hidden rows, the six linears, the indexer, the top-k, sparse_sdpa, the out linear
and its reduce-scatter) is ~62 small programs per layer.  This family replaces them program by program, each bitwise
against the chain it stands for on the rows the tile carries (rows past R are not zero on the verify tile: every
program masks or ignores them exactly as its chain does).  Opt-in as one name, ``QWEN38_FUSED=qsa_rows``.

Program 1, ``score_blocks_rows``: the indexer scores' all-reduce and mask add.  The chain is ``ttnn.all_reduce`` (its
composite: the all-broadcast, a concat, tilize, ``moreh_sum`` over the device dim, untilize) and ``ttnn.add`` of the
per-row block mask, six programs and ~130 us per layer at 32 rows.  Here, as the decode chain's ``qsa_score_merge``
does for one row: the local rows as 2 KB pages, one ``all_gather`` (device d's page c of row r lands at page
``(d * rows + r) * chunks + c``), then ``qsa_block.score_merge`` (the composite's moreh_sum order, then the mask add)
over all 32 rows of the tile at once -- that program's device test holds it bitwise against the chain per row at 1, 2,
5 and 32 rows; the mask is the chunk's ``[1, 1, 32, blocks]`` row mask, so the rows past R are masked as the chain
masks them.
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
    """The local partial block scores ``[1, 1, rows, blocks]`` bf16 ROW_MAJOR of the tile's rows, summed over the
    ``cluster_axis`` devices and masked with ``mask`` (the same shape): the pages all-gathered, the fused merge over
    the rows.  Returns the masked scores ``[1, 1, rows, blocks]`` bf16 ROW_MAJOR (the top-k's input)."""

    rows, blocks = _score_rows_shape(score_rows, mask)
    pages = blocks // PAGE
    # row r's chunk c at page r * pages + c; the gather stacks the devices' page runs, device-major
    paged = ttnn.reshape(score_rows, (1, 1, rows * pages, PAGE))
    return _gather_and_merge(paged, rows, pages, mask, cluster_axis)


def score_blocks_rows_from_scores(local_scores, mask, *, cluster_axis: int, pages):
    """Program 1 fed by program 4: the indexer's local score rows ``[1, 1, rows, W >= blocks]`` (the resident blocks
    and the cache's spare tile of columns) repaged by ``pages`` (:func:`score_pages`, or its composed slice + reshape)
    straight into the gather's 2 KB pages, then the same all-gather and fused merge as :func:`score_blocks_rows`."""

    rows, blocks = _mask_shape(mask)
    paged = pages(local_scores, blocks)
    return _gather_and_merge(paged, rows, blocks // PAGE, mask, cluster_axis)


def _gather_and_merge(paged, rows: int, pages: int, mask, cluster_axis: int):
    dram = ttnn.DRAM_MEMORY_CONFIG
    gathered = ttnn.all_gather(paged, dim=2, cluster_axis=cluster_axis, memory_config=dram)
    ttnn.deallocate(paged)
    if tuple(gathered.shape) != (1, 1, DEVICES * rows * pages, PAGE):
        raise RuntimeError(
            f"gathered score pages have shape {tuple(gathered.shape)}, expected [1, 1, {DEVICES * rows * pages}, {PAGE}]"
        )
    masked = qsa_block.score_merge(gathered, mask)
    ttnn.deallocate(gathered)
    return masked


def _mask_shape(mask):
    mshape = tuple(mask.shape)
    if (
        len(mshape) != 4
        or mshape[:2] != (1, 1)
        or not 1 <= mshape[2] <= ttnn.TILE_SIZE
        or mshape[3] % PAGE
        or mask.dtype != ttnn.bfloat16
        or mask.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"the block mask must be ROW_MAJOR bf16 [1, 1, 1..32, k * {PAGE}], got {mask.layout} {mshape}")
    return mshape[2], mshape[3]


# --- program 4: the indexer's rows as the gather's pages --------------------------------------------------------------

PAGES_NAME = "qsa_score_pages"
SCORE_PAGES = fp.kernel_source(NAME, "score_pages.cpp")
PAGES_BATCH = 4  # pages per read / write round on a core (8 KB of L1)


def score_pages_admits(local_scores, blocks) -> bool:
    """Program 4's input contract as a predicate: one row tile of score rows ``[1, 1, 1..32, W >= blocks]`` bf16
    ROW_MAJOR with ``blocks`` a positive multiple of the page; anything else (the 128-row chunk's four tiles, the slab)
    keeps the chain's slice + reshape."""

    shape = tuple(local_scores.shape)
    return (
        len(shape) == 4
        and shape[:2] == (1, 1)
        and 1 <= shape[2] <= ttnn.TILE_SIZE
        and local_scores.dtype == ttnn.bfloat16
        and local_scores.layout == ttnn.ROW_MAJOR_LAYOUT
        and type(blocks) is int
        and blocks > 0
        and blocks % PAGE == 0
        and blocks <= shape[3]
    )


def score_pages(local_scores, blocks: int):
    """Program 4: the indexer's local score rows ``[1, 1, rows, W]`` bf16 ROW_MAJOR with ``W >= blocks`` (the chain
    slices the ``blocks`` resident columns off the cache's spare tile) repaged as the all-gather's pages
    ``[1, 1, rows * blocks // 1024, 1024]``: row r's chunk c at page ``r * chunks + c``.  One data-movement program
    (pure 2 KB copies: the bytes are the chain's slice + reshape) in place of those two programs."""

    shape = tuple(local_scores.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or not 1 <= shape[2] <= ttnn.TILE_SIZE
        or local_scores.dtype != ttnn.bfloat16
        or local_scores.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"score rows must be ROW_MAJOR bf16 [1, 1, 1..32, W], got {local_scores.layout} {shape}")
    if type(blocks) is not int or blocks <= 0 or blocks % PAGE or blocks > shape[3]:
        raise ValueError(
            f"blocks must be a positive multiple of {PAGE} within the rows' width {shape[3]}, got {blocks!r}"
        )
    rows, chunks = shape[2], blocks // PAGE
    mesh = local_scores.device()
    out = fp.allocate((1, 1, rows * chunks, PAGE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh)
    work = fp.split_work(rows * chunks, mesh)
    cores = fp.core_rectangle(work, mesh)
    page_bytes = fp.TILE_BYTES[ttnn.bfloat16]  # 1024 bf16 = one 2 KB page
    cta = [chunks, PAGES_BATCH, *fp.accessor_args(local_scores), *fp.accessor_args(out)]
    cbs = [fp.cb_descriptor(0, ttnn.bfloat16, page_bytes, PAGES_BATCH, cores)]
    kernel = fp.reader_kernel(
        SCORE_PAGES,
        cores,
        cta,
        [(w.core, [local_scores.buffer_address(), out.buffer_address(), w.start, w.count]) for w in work],
    )
    # the resident columns of the rows read once, the pages written once; the pages keep the rows' placement
    meta = fp.program_meta(
        NAME,
        "score_pages",
        rows,
        partial=((local_scores, rows * blocks * 2),),
        writes=(out,),
        cores=len(work),
        outputs=((out, local_scores),),
    )
    return fp.run_program([local_scores, out], fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def score_pages_composed(local_scores, blocks: int):
    """The chain: ``ttnn.slice`` to the resident blocks, then the row-major reshape into the gather's pages."""

    rows = int(local_scores.shape[2])
    dram = ttnn.DRAM_MEMORY_CONFIG
    sliced = ttnn.slice(local_scores, (0, 0, 0, 0), (1, 1, rows, blocks), memory_config=dram)
    paged = ttnn.reshape(sliced, (1, 1, rows * (blocks // PAGE), PAGE))
    if paged.buffer_address() != sliced.buffer_address():
        ttnn.deallocate(sliced)
    return paged


def score_blocks_rows_composed(score_rows, mask, *, cluster_axis: int, topology=None):
    """The chain: ``ttnn.all_reduce`` over the devices, then the mask add (``fast_and_approximate_mode=False``)."""

    _score_rows_shape(score_rows, mask)
    dram = ttnn.DRAM_MEMORY_CONFIG
    kwargs = {} if topology is None else {"topology": topology}
    scores = ttnn.all_reduce(score_rows, cluster_axis=cluster_axis, memory_config=dram, **kwargs)
    masked = ttnn.add(scores, mask, memory_config=dram, fast_and_approximate_mode=False)
    ttnn.deallocate(scores)
    return masked


def selection_rows(block_ids, sentinel_pad, block_offsets_rows, row_keep_bits, row_fill):
    """Program 3: the verify tile's sparse-attention rows of token ids [1, 1, 32, 2080] uint32 from the top-k block ids
    [1, 1, 32, 512] and the pass's keep / fill rows -- the decode ``qsa_selection_row`` program on the 32 rows (its
    per-row integer chain: shift, repeat, offset add, sentinel concat, keep and, fill or, exact), with the module's
    one-row ``sentinel_pad`` and the chunk constants' per-row ``block_offsets_rows`` [1, 1, 32, 2048]."""

    shape = tuple(block_ids.shape)
    if shape != (1, 1, fp.TILE, qsa_block.BLOCK_IDS):
        raise ValueError(
            f"the verify selection takes the 32-row block ids [1, 1, 32, {qsa_block.BLOCK_IDS}], got {shape}"
        )
    return qsa_block.selection_row(block_ids, sentinel_pad, block_offsets_rows, row_keep_bits, row_fill)


def selection_rows_composed(block_ids, sentinel_pad_rows, block_offsets_rows, row_keep_bits, row_fill):
    """The chain (ttnn/qsa.py ``_materialize_rows_chunk`` after ``topk_large_indices``) on the 32-row tile, with the
    chunk constants' per-row ``sentinel_pad_rows`` [1, 1, 32, 32]."""

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
        gate=None,  # the rows micro-test and the pass pair (the MTP lead's gate) stand for the family
    )
)

register(
    FusedKernel(
        name=PAGES_NAME,
        replaces=(
            "the QSA verify tile's score slice to the resident blocks and the row-major reshape into the all-gather's "
            "2 KB pages (2 programs per attention layer, verify and draft) as one data-movement program (qsa_rows program 4)"
        ),
        tolerance=BITWISE,
        fused=score_pages,
        composed=score_pages_composed,
        admits=score_pages_admits,
        # pure copies: the die test and the line test (bitwise the chain) are the gate.  Number of record, the pass pair
        # of 2026-09-26 on the 1x4 p150 line (greedy fused, json / prose / chat560): verify replay -0.18 / -0.15 / -0.18 ms
        # per pass, tokens per pass identical; default since.
        gate=None,
    )
)

# program 2: the main tail with the verify rows' KV stage (the family's attribute is the function; the module stays
# importable as ttnn.fused.qsa_rows.main_tail_rows through sys.modules)
from .main_tail_rows import kv_stage, main_tail_rows, main_tail_rows_composed  # noqa: E402

# program 5: the post-attention glue on the tile as one 48-core program (default, qsa_rows_post_attention); the
# kernels, the input contract and the composed chain live in post_attention_rows.py, the launch here
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
    """Program 5's input contract as a predicate: the qg projection as one row tile ``[1, 1, 1..32, 3072]`` (the verify
    tile; the one-row draft) and sparse_sdpa's ROW_MAJOR ``[1, 32, rows, 256]``; the chunk body's six-branch gate
    ``[1, 6, rows, 256]``, the 128-row chunk and the slab keep the chain."""

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
    """sigmoid(gate) x attention for the six local heads over the tile's rows -> the head-major ``[1, 1, rows, 1536]``
    bf16 TILE row in ``memory_config`` (the out-projection's activation shard).  ``attention`` is sparse_sdpa's
    ROW_MAJOR ``[1, 32, rows, 256]``; the gates are the second 256 columns of each head's 512 in ``qg_ws`` from tile
    ``qg_first``.  One core per (head, tile column)."""

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
    # the six local heads' attention rows and gate tiles in, the head-major row out (head-sharded on dim 3)
    meta = fp.program_meta(
        NAME,
        "post_attention_rows",
        rows,
        partial=(
            (attention, qsa_block.LOCAL_HEADS * rows * qsa_block.HEAD_DIM * 2),
            (qg_ws, qsa_block.LOCAL_HEADS * qsa_block.HEAD_TILES * fp.TILE_BYTES[ttnn.bfloat16]),
        ),
        writes=(out,),
        flops=2 * rows * qsa_block.OUT_WIDTH,
        cores=len(work),
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
        # the device test (bitwise the chain on the 1x4 line) is the gate.  Number of record, the pass pair of 2026-09-26
        # on the 1x4 p150 line (greedy fused, json / prose / chat560): verify replay -0.35 / -0.38 / -0.36 ms and draft
        # replay -0.10 ms per pass, pass wall -0.36 / -0.44 / -0.31 ms, tokens per pass identical; default since.
        gate=None,
    )
)


__all__ = [
    "NAME",
    "PAGES_NAME",
    "score_pages_admits",
    "post_attention_rows_admits",
    "PA_NAME",
    "post_attention_rows",
    "post_attention_rows_composed",
    "score_pages",
    "score_pages_composed",
    "score_blocks_rows_from_scores",
    "kv_stage",
    "main_tail_rows",
    "main_tail_rows_composed",
    "score_blocks_rows",
    "score_blocks_rows_composed",
    "selection_rows",
    "selection_rows_composed",
]
