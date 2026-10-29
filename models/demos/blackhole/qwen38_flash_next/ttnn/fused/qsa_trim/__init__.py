# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``qsa_trim`` (QWEN38_QSA_TRIM=1, default off): glue cuts of the one-row QSA decode layer, each a program that already
exists for the verify tile (``qsa_rows``) used on the decode row, plus the new ones of this package.  Every cut is
BITWISE against the program(s) it replaces (pure data movement, or the same arithmetic on the same operands):

  score_pages      ttnn.slice (1.5 us) + ttnn.reshape (4.0 us) + 2 launch gaps  ->  one data-movement program
  post_attention   qsa_block.post_attention (3 cores, 16.5 us)  ->  qsa_rows.post_attention_rows (one core per head x tile
                   column, 48 cores)
  reduce_w2n       (opt-in, QWEN38_QSA_TRIM_REDUCE=1) widen_partial + row_reduce + typecast -> see ``reduce_w2n``

``QWEN38_QSA_TRIM_OPTS`` (study): comma list of ``pages``, ``post``, ``reduce`` replacing the default set.
"""

from __future__ import annotations

import os

import ttnn

from .. import program as fp
from .. import qsa_rows

from .. import gr_fold
from .. import row_reduce as rr
from ...contracts import TP_SIZE

RR_COMPUTE = fp.kernel_source("qsa_trim", "rr_compute.cpp")
CB_IN, PACK_CB, CB_FS, CB_BS, CB_F, CB_B, CB_G, CB_OUT = (
    rr.CB_IN,
    rr.PACK_CB,
    rr.CB_FS,
    rr.CB_BS,
    rr.CB_F,
    rr.CB_B,
    rr.CB_G,
    rr.CB_OUT,
)
ENV = "QWEN38_QSA_TRIM"
OPTS_ENV = "QWEN38_QSA_TRIM_OPTS"
NAME = "qsa_trim"
DEFAULT_OPTS = ("pages", "post", "reduce")


def options(environ=os.environ) -> tuple[str, ...]:
    if environ.get(ENV, "1") != "1":  # default on (x45 A/B -0.23 ms)
        return ()
    listed = environ.get(OPTS_ENV)
    return DEFAULT_OPTS if listed is None else tuple(o for o in listed.split(",") if o)


def enabled(environ=os.environ) -> bool:
    return bool(options(environ))


def reduce_w2n_admits(partial, *, cluster_axis=1) -> bool:
    """One valid row of a bf16 TILE partial on the TP line (the out-projection's L1 shard), v2 row_reduce on."""

    try:
        shape = tuple(int(v) for v in partial.shape)
        padded = tuple(int(v) for v in partial.padded_shape)
        return (
            rr.enabled_by_env()
            and rr.version() == 2
            and cluster_axis == gr_fold.TP_AXIS
            and len(shape) == 4
            and shape[:2] == (1, 1)
            and shape[2] == 1
            and partial.dtype == ttnn.bfloat16
            and partial.layout == ttnn.TILE_LAYOUT
            and padded[2] == fp.TILE
            and padded[3] == 80 * fp.TILE  # the landing buffer of prepare2(mesh, 80)
            and tuple(int(v) for v in partial.device().shape) == (1, TP_SIZE)
            and fp.in_l1(partial)
            and rr._payload() >= 16
        )
    except Exception:
        return False


def reduce_w2n(partial, *, reference):
    """widen_partial + row_reduce (v2, fp32) + the typecast to bf16, as one program: the bf16 partial ``[1, 1, 1, 2560]``
    TILE (L1 shard) -> the bf16 TILE ``[1, 1, 1, 320]`` DRAM output of the typecast.  The transport is row_reduce v2's
    (transport2.cpp, unchanged: bf16 rows in, bf16 rows out, the landing buffer and counter shared with every other
    v2 call of the mesh); the compute is ``kernels/rr_compute.cpp``.  ``reference``: the tensor whose mesh topology
    the output takes (replicated; sharded on dim 3)."""

    mesh = partial.device()
    shape = tuple(int(v) for v in partial.shape)
    rows, tiles, out_tiles, chunk, _packed, _slot, _scratch = rr._geometry(partial)
    state = rr.prepare2(mesh, tiles)
    tile_bytes = fp.TILE_BYTES[ttnn.bfloat16]
    pitch = chunk  # an L1 partial: rows are read at their own size
    out = fp.allocate((1, 1, rows, shape[3] // TP_SIZE), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    total = tiles * 2 * pitch
    payload = rr._payload()
    transports = gr_fold.TRANSPORT["partials"][: rr.LINKS]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    spec = gr_fold.Transport(
        out,
        partial,
        CB_IN,
        tiles,
        gr_fold.SOURCE_TENSOR,
        rr.PHASE2,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=rr.TRANSPORT2_KERNEL,
        extra_args=(
            rows,
            PACK_CB,
            pitch,
            CB_FS,
            CB_BS,
            CB_F,
            CB_B,
            CB_G,
            CB_OUT,
            out_tiles,
            rr.MIRROR,
            state["land_addr"],
            state["data1"],
            payload,
            state["slot"],
        ),
    )
    bf16, fp32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        fp.cb_descriptor(CB_IN, bf16, tile_bytes, TP_SIZE, cores),
        fp.cb_descriptor(PACK_CB, bf16, total, 1, cores),
        fp.cb_descriptor(CB_FS, bf16, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_BS, bf16, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_F, fp32, fp.TILE_BYTES[fp32], 2, cores),
        fp.cb_descriptor(CB_B, fp32, fp.TILE_BYTES[fp32], 2, cores),
        fp.cb_descriptor(CB_G, fp32, fp.TILE_BYTES[fp32], 2, cores),
        fp.cb_descriptor(CB_OUT, bf16, tile_bytes, 1, cores),
    ]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]

    def compute(rank):
        logical = TP_SIZE - 1 - rank if rr.MIRROR else rank
        return [
            fp.compute_kernel(
                RR_COMPUTE,
                cores,
                [CB_IN, CB_FS, CB_BS, CB_F, CB_B, CB_G, CB_OUT, 1, TP_SIZE],
                [(transports[0], [logical])],
                fp32_dest=True,
            )
        ]

    meta = fp.program_meta(
        NAME,
        "reduce_w2n",
        rows,
        reads=(partial,),
        writes=(out,),
        l1_bytes=total + (TP_SIZE - 1) * total,
        flops=(TP_SIZE - 1) * out_tiles * fp.TILE * fp.TILE,
        cores=rr.LINKS,
        outputs=((out, 3),),
    )
    fp.run_program(
        [partial, out],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=compute, links=rr.LINKS),
        meta=meta,
    )
    return fp.stamp_topology(out, reference, 3)


def score_pages_row0(local_scores, blocks: int):
    """The decode's slice of row 0 to the resident blocks plus the 2 KB repage as ONE program: qsa_rows.score_pages'
    kernel and launch for ONE row of the indexer's 32-row score tile ``[1, 1, 32, W >= blocks]`` (the verify form
    repages every row of its tile).  Returns ``[1, 1, blocks // 1024, 1024]`` bf16 ROW_MAJOR; the bytes are the chain's.
    """

    chunks = blocks // qsa_rows.PAGE
    mesh = local_scores.device()
    out = fp.allocate((1, 1, chunks, qsa_rows.PAGE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh)
    work = fp.split_work(chunks, mesh)
    cores = fp.core_rectangle(work, mesh)
    page_bytes = fp.TILE_BYTES[ttnn.bfloat16]
    cta = [chunks, qsa_rows.PAGES_BATCH, *fp.accessor_args(local_scores), *fp.accessor_args(out)]
    cbs = [fp.cb_descriptor(0, ttnn.bfloat16, page_bytes, qsa_rows.PAGES_BATCH, cores)]
    kernel = fp.reader_kernel(
        qsa_rows.SCORE_PAGES,
        cores,
        cta,
        [(w.core, [local_scores.buffer_address(), out.buffer_address(), w.start, w.count]) for w in work],
    )
    meta = fp.program_meta(
        NAME,
        "score_pages_row0",
        1,
        partial=((local_scores, blocks * 2),),
        writes=(out,),
        cores=len(work),
        outputs=((out, local_scores),),
    )
    return fp.run_program([local_scores, out], fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def score_pages_row0_admits(local_scores, blocks) -> bool:
    return qsa_rows.score_pages_admits(local_scores, blocks)


def reduce_for(qsa):
    """``reduce_w2n`` (with ``.admits``) when 'reduce' is on, else None."""

    if "reduce" not in options():
        return None
    reduce_w2n.admits = reduce_w2n_admits
    return reduce_w2n


def score_pages_for(qsa):
    """``score_pages_row0(local_scores, blocks)`` when 'pages' is on, else None."""

    if "pages" not in options():
        return None
    score_pages_row0.admits = score_pages_row0_admits  # the caller keeps the slice + reshape chain outside the contract
    return score_pages_row0


def post_attention_for(qsa):
    """The 48-core ``post_attention_rows`` with the decode program's call signature when 'post' is on, else None."""

    if "post" not in options():
        return None

    run = qsa_rows.post_attention_rows
    run.admits = qsa_rows.post_attention_rows_admits
    return run
