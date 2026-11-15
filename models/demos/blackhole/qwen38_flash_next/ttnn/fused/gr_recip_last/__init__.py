# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gr_recip_last``: the GR read's front with the rsqrt applied last, as one program on ``gr_fold``'s line transports.

The norm cores multicast u_b = bf16(x_b * gamma_b/4) right after the stats, so the workers' down matmul (one dest tile
per residual stream) runs while the stats exchange is in flight; they then multicast recip_b = rsqrt(mean_b + eps)
(bitwise the chain's) and the workers form P = sum_b P_b * recip_b. The matmul has no spill and the gate's operand
bf16(u_b * recip_b) one bf16 rounding where the chain has two (``docs/NUMERICS.md``), hence COMPONENT class. Needs
``gr_read``; ``QWEN38_FUSED_OFF=gr_recip_last`` restores ``gr_read_fused``.
"""

from __future__ import annotations

import ttnn

from .. import gr_fold, gr_read
from .. import program as fp
from ..registry import COMPONENT, FusedKernel, register

NAME = "gr_recip_last"
TP_SIZE = gr_fold.TP_SIZE
STATS_U_RECIP = fp.kernel_source(NAME, "stats_u_recip_compute.cpp")
DOWN_STREAMS = fp.kernel_source(NAME, "down_streams_compute.cpp")
MCAST_READER2 = fp.kernel_source(NAME, "mcast_reader2.cpp")
MCAST_WRITER3 = fp.kernel_source(NAME, "mcast_writer3.cpp")
KERNELS = {
    "stats_u_recip_compute": STATS_U_RECIP,
    "down_streams_compute": DOWN_STREAMS,
    "mcast_reader2": MCAST_READER2,
    "mcast_writer3": MCAST_WRITER3,
}
RECIP_READY = 6  # program-local semaphore on the workers: the four recip tiles landed
# the norm cores' CBs: gr_read.normalize_down's 0-7 (0 residual, 1 gathered stats, 2 scaler, 3 eps, 4 gamma, 5 var,
# 6 recip, 7 the kept u) plus these
CB_U = 16  # u row, the multicast source (bf16, 20)
CB_NOUT = 18  # normalized' for the gate (bf16, 20)
CB_RECIP_OUT = 19  # the recip tile for the workers (fp32, 1)
CB_SSCALER, CB_X2, CB_SOUT, CB_SSCRATCH = gr_fold.FRONT_STATS_CBS  # 20-23: the stats phase's
# the workers' CBs: gr_read.normalize_down's 8 (u row landing), 9 (weight column), 10 (interm), 17 (out) plus these
CB_IN0, CB_W, CB_INTERM, CB_OUT = 8, 9, 10, 17
CB_RECIP_W = 12  # the four recip tiles landing (fp32)
CB_PROD = 13  # the four scaled partials (fp32)
CB_ZERO = 14  # the fold's zero tile (fp32)
WEIGHT_BATCH = 8  # weight tiles per push, as gr_read.normalize_down
U_BLOCK = 4  # tiles per fp32 dest block on the norm cores


def _mcast_reader2(cores, streams, runtime, *, recv, recv2, zero_cb=gr_read.NONE_CB):
    """``recv`` / ``recv2`` = (cb, tiles, senders, semaphore id); streams: [(tensor, cb)] (<= 2); runtime: per core ->
    stream arg lists."""

    compile_args = [*recv, *recv2, len(streams), *[cb for _, cb in streams], *[0] * (2 - len(streams)), zero_cb]
    for slot in range(2):
        compile_args += fp.accessor_args(streams[min(slot, len(streams) - 1)][0])
    return fp.reader_kernel(
        MCAST_READER2,
        cores,
        compile_args,
        [(core, [a for args in stream_args for a in args]) for core, stream_args in runtime],
    )


def stats_recip_down_gather(residual, norm_scale, down_inject, *, links: int | None = None):
    """The front as one program on normalize_down's cores and gr_fold's transports.

    Returns ``(gathered_stats, normalized', gathered_partials)`` in the chain's gathered shapes and pages."""

    rows = gr_read.residual_rows(residual)
    gr_read._expect(
        norm_scale, (1, gr_read.BRANCHES, fp.TILE, gr_read.LOCAL_HIDDEN), gr_read.FP32, "GR norm_scale rows"
    )
    gr_read._expect(down_inject, (1, 1, gr_read.FLAT_WIDTH, gr_read.PARTIAL_WIDTH), gr_read.BF16, "GR down_inject")
    mesh = residual.device()
    transports = gr_fold.TRANSPORT["partials"]
    links = min(len(transports), gr_fold.line(mesh).links) if links is None else links
    transports = transports[:links]
    gathered_stats = fp.allocate((1, gr_read.BRANCHES, rows, fp.TILE * TP_SIZE), gr_read.BF16, ttnn.TILE_LAYOUT, mesh)
    normalized = fp.allocate((1, 1, rows, gr_read.FLAT_WIDTH), gr_read.BF16, ttnn.TILE_LAYOUT, mesh)
    gathered = fp.allocate((TP_SIZE, 1, rows, gr_read.PARTIAL_WIDTH), gr_read.FP32, ttnn.TILE_LAYOUT, mesh)
    n_tiles = gr_read._tiles_wide(down_inject)
    workers, producers, rect = gr_read._rectangle(mesh, 6, 2, rows_above=gr_read.BRANCHES)
    if any(c in transports for c in workers + producers):
        raise RuntimeError(f"the transport cores {transports} overlap the front's cores")
    noc = gr_read.noc_map(mesh)
    w_set, p_set = gr_read._core_set(workers), gr_read._core_set(producers)
    pw_set = gr_read._core_set(workers + producers)
    ps_set = gr_read._core_set(producers + list(transports))
    wt_set = gr_read._core_set(workers + list(transports))
    T_BF16, T_FP32, HT, ST, FT, PT = (
        gr_read.TILE_BF16,
        gr_read.TILE_FP32,
        gr_read.HIDDEN_TILES,
        gr_read.STATS_TILES,
        gr_read.FLAT_TILES,
        gr_read.PARTIAL_TILES,
    )
    B = gr_read.BRANCHES
    cbs = [
        fp.cb_descriptor(0, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(1, gr_read.BF16, T_BF16, ST, p_set),
        fp.cb_descriptor(2, gr_read.BF16, T_BF16, 1, p_set),
        fp.cb_descriptor(3, gr_read.BF16, T_BF16, 1, p_set),
        fp.cb_descriptor(4, gr_read.FP32, T_FP32, HT, p_set),
        fp.cb_descriptor(5, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(6, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(7, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(CB_U, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(CB_NOUT, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(CB_RECIP_OUT, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(CB_SSCALER, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(CB_X2, gr_read.FP32, T_FP32, HT, p_set),
        fp.cb_descriptor(CB_SOUT, gr_read.BF16, T_BF16, 1, p_set),
        fp.cb_descriptor(CB_SSCRATCH, gr_read.BF16, T_BF16, B, ps_set),
        fp.cb_descriptor(CB_IN0, gr_read.BF16, T_BF16, FT, pw_set),  # the u row lands here (one address on every core)
        fp.cb_descriptor(CB_RECIP_W, gr_read.FP32, T_FP32, B, pw_set),  # the recip tiles land here
        fp.cb_descriptor(CB_W, gr_read.BF16, T_BF16, FT, w_set),
        fp.cb_descriptor(CB_INTERM, gr_read.FP32, T_FP32, B, w_set),
        fp.cb_descriptor(CB_PROD, gr_read.FP32, T_FP32, B, w_set),
        fp.cb_descriptor(CB_ZERO, gr_read.FP32, T_FP32, 1, w_set),
        fp.cb_descriptor(CB_OUT, gr_read.FP32, T_FP32, 1, w_set),
        fp.cb_descriptor(gr_fold.PARTIALS_SCRATCH_CB, gr_read.FP32, T_FP32, PT, wt_set),
    ]
    # the norm core's reader: the residual once (the stats and u phases share it), gamma, then the gathered stats once
    # the stats transport and the core's own writer signalled (the gate: links + 1)
    reader = gr_read._reader(
        p_set,
        [(residual, 0), (norm_scale, 4), (gathered_stats, 1)],
        [(gr_read.CONST_SCALER, CB_SSCALER), (gr_read.CONST_SCALER, 2), (gr_read.CONST_COL_SCALAR, 3)],
        [
            (
                core,
                (
                    [
                        gr_read._stream(residual, 1, HT, b * HT, 1, 0, 4),
                        gr_read._stream(norm_scale, 1, HT, b * HT, 1, 0, 4),
                        gr_read._stream(gathered_stats, 1, ST, b * ST, 1, 0, ST),
                    ],
                    [gr_read._bits(1.0), gr_read.AVG_SCALER_BITS, gr_read._bits(gr_read.EPS)],
                ),
            )
            for b, core in enumerate(producers)
        ],
        gate=(2, gr_fold.FRONT_STATS_READY, links + 1),
    )
    compute = fp.compute_kernel(
        STATS_U_RECIP,
        p_set,
        [HT, ST, U_BLOCK, CB_U, CB_SSCALER, CB_X2, CB_SOUT, CB_RECIP_OUT, CB_NOUT],
        fp32_dest=True,
    )
    receiver = _mcast_reader2(
        w_set,
        [(down_inject, CB_W)],
        [(core, [gr_read._stream(down_inject, 1, FT, w, n_tiles, 0, WEIGHT_BATCH)]) for w, core in enumerate(workers)],
        recv=(CB_IN0, FT, B, 0),
        recv2=(CB_RECIP_W, B, B, RECIP_READY),
        zero_cb=CB_ZERO,
    )
    down = fp.compute_kernel(
        DOWN_STREAMS,
        w_set,
        [FT, WEIGHT_BATCH, HT, CB_IN0, CB_W, CB_INTERM, CB_RECIP_W, CB_PROD, CB_ZERO, CB_OUT, B],
        fp32_dest=True,
    )
    p_go, p_scratch, p_done = gr_fold.PARTIALS_SEMAPHORES
    s_scratch = gr_fold.FRONT_STATS_SCRATCH_SEM
    accessors = (
        fp.accessor_args(gathered_stats)
        + fp.accessor_args(gathered_stats)
        + fp.accessor_args(normalized)
        + fp.accessor_args(normalized)
        + fp.accessor_args(normalized)
        + fp.accessor_args(normalized)
    )

    def kernels(rank):
        # the norm cores' three-phase writer: A = the stats tile into transport core (b % links)'s scratch slot
        # b // links and the device's own page 4b + rank, then the core's own gate signal; B = the u row into the
        # workers' CB at tile offset b * HT; C = the recip tile into the workers' recip CB at offset b, and the
        # normalized' row written to the normalized tensor (pages b * HT ..) as the phase's extra stream
        runtime = []
        for b, core in enumerate(producers):
            x, y = noc[(transports[b % links].x, transports[b % links].y)]
            phase_a = [b // links, x, y, x, y, gathered_stats.buffer_address(), b * TP_SIZE + rank, 1, 0, 0, 0, 1, 1]
            phase_b = [b * HT, *rect, 0, 0, 0, 0, 0, 0, 1, 1]
            phase_c = [b, *rect, 0, 0, 0, normalized.buffer_address(), HT, b * HT, 1, 4]
            runtime.append((core, phase_a + phase_b + phase_c + list(noc[(core.x, core.y)])))
        writer3 = fp.writer_kernel(
            MCAST_WRITER3,
            p_set,
            [CB_SOUT, CB_SSCRATCH, 1, 1, gr_read.NONE_CB, s_scratch]
            + [CB_U, CB_IN0, HT, 0, gr_read.NONE_CB, 0]
            + [CB_RECIP_OUT, CB_RECIP_W, 1, 0, CB_NOUT, RECIP_READY]
            + [gr_fold.FRONT_STATS_READY]
            + accessors,
            runtime,
        )
        # the workers' writer: the partial tile into transport core (w % links)'s partials scratch slot w // links and
        # the device's own page 12 * rank + w
        runtime = []
        for w, core in enumerate(workers):
            x, y = noc[(transports[w % links].x, transports[w % links].y)]
            runtime.append((core, (w // links, (x, y, x, y), (PT * rank + w, 1), (0, 0, 1, 1))))
        writer = gr_read._mcast_writer(
            w_set,
            runtime,
            src_cb=CB_OUT,
            dst_cb=gr_fold.PARTIALS_SCRATCH_CB,
            tiles=1,
            tiles_tensor=gathered,
            sem=p_scratch,
        )
        return [reader, compute, writer3, receiver, down, writer]

    semaphores = (
        [fp.semaphore_descriptor(0, pw_set)]
        + [fp.semaphore_descriptor(sem, wt_set) for sem in gr_fold.PARTIALS_SEMAPHORES]
        + [
            fp.semaphore_descriptor(s_scratch, ps_set),
            fp.semaphore_descriptor(gr_fold.FRONT_STATS_READY, p_set),
            fp.semaphore_descriptor(RECIP_READY, pw_set),
        ]
    )
    phases = [
        gr_fold.Transport(
            gathered_stats,
            gathered_stats,
            CB_SSCRATCH,
            B,
            gr_fold.SOURCE_PRODUCERS,
            "stats",
            transports,
            semaphore_ids=(p_go, s_scratch, p_done),
            consumers=tuple(noc[(c.x, c.y)] for c in producers),
            consumer_sem=gr_fold.FRONT_STATS_READY,
        ),
        gr_fold.Transport(
            gathered,
            gathered,
            gr_fold.PARTIALS_SCRATCH_CB,
            PT,
            gr_fold.SOURCE_PRODUCERS,
            "partials",
            transports,
            semaphore_ids=gr_fold.PARTIALS_SEMAPHORES,
            page_strides=(1, PT),
        ),
    ]
    meta = fp.program_meta(
        NAME,
        "stats_recip_down_gather",
        outputs=((gathered_stats, None), (normalized, 3), (gathered, None)),
    )
    fp.run_program(
        [residual, norm_scale, down_inject, gathered_stats, normalized, gathered],
        gr_fold.transport_mesh_program(mesh, phases, semaphores=semaphores, cbs=cbs, kernels=kernels, links=links),
        meta=meta,
    )
    return gathered_stats, normalized, gathered


def read_front(residual, gamma_rows, down_inject):
    """``gr_read_fused``'s ``read_front`` hook: the front above; the gathered stats stay internal."""

    gathered_stats, normalized, gathered_partials = stats_recip_down_gather(residual, gamma_rows, down_inject)
    ttnn.deallocate(gathered_stats)
    return normalized, gathered_partials


def gr_read_recip_last(module, residual):
    """``gr_read``'s read with this front (two programs per read: the front and ``low_rank_gate``)."""

    return gr_read.gr_read_fused(module, residual, read_front=read_front)


register(
    FusedKernel(
        name=NAME,
        replaces="gr_read's stats, all_gather, normalize_down and all_gather_async as ONE program on the line "
        "transports with the rsqrt applied after the down projection, so the u multicast and the matmul run under the "
        "stats round trip; COMPONENT class, on by default, resolved by the GR module when gr_read is on",
        tolerance=COMPONENT,
        fused=gr_read_recip_last,
        composed=gr_read.gr_read_fused,
        gate=None,  # needs the TP4 line; checked by the mesh replica against the oracle
        component_proof="the mesh replica's block input against the CPU oracle on captured residual rows",
    )
)
