# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GR read's front with the rsqrt applied LAST (``gr_recip_last``): gr_fold's one-program front re-associated so
that nothing waits on the statistics exchange.

gr_fold's ``stats_normalize_down_gather`` chains stats -> line gather -> rsqrt -> normalize -> multicast -> down matmul ->
line gather, and the measured 5-row timeline (2026-09-27) puts 22 of its 44 us between the two gathers waiting on
rsqrt(mean): the norm compute, the 80-tile multicast into the workers and the matmul all need the gathered stats.
Here the norm cores compute u_b = bf16(x_b * gamma_b/4) right after the stats (no statistics needed) and multicast
it at once; the workers run the same 80 matmul blocks into four dest tiles, one per residual stream, while the stats
round trip is in flight; the norm cores turn the gathered stats into recip_b = rsqrt(mean_b + eps) (the chain's
reduce / eps / rsqrt, bitwise that far) and multicast the four fp32 recip tiles to the workers, which scale each
stream's partial and sum the four (P = sum_b P_b * recip_b) into the partial tile the partials gather carries as
today.  The gate's operand normalized' = bf16(u_b * recip_b) is written by the norm cores off the critical path.
The transport kernels, the partials payload and ``low_rank_gate`` are gr_fold's / gr_read's unchanged.

Rounding points that move against the chain (``docs/NUMERICS.md``): chain = bf16(bf16(x * rsqrt) * gamma/4) into
the matmul, the DRAM-sharded matmul's spill/reload every 8 K tiles, then the device sum; here bf16(x * gamma/4)
into the matmul (no spill), rsqrt onto the fp32 partial, then the unchanged device sum -> bf16 -> silu; the gate's
operand has one bf16 rounding where the chain has two.  COMPONENT class, on by default since 2026-09-27 with the
component proof recorded in its registry entry (the 1x4 replica against the tt/ oracle, a dev tool, and the paired served
sample; ``docs/NUMERICS.md``); it needs ``gr_read``; ``QWEN38_FUSED_OFF=gr_recip_last`` restores the fold's front.
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
RECIP_READY = 6  # program-local semaphore on the workers (and the producers that raise it): the four recip tiles landed
# the norm cores' CBs: gr_read.normalize's 0-7 (0 residual, 1 gathered stats, 2 scaler, 3 eps, 4 gamma, 5 var,
# 6 recip, 7 the kept u) plus these
CB_U = 16  # u row, the multicast source (bf16, 20)
CB_NOUT = 18  # normalized' for the gate (bf16, 20)
CB_RECIP_OUT = 19  # the recip tile for the workers (fp32, 1)
CB_SSCALER, CB_X2, CB_SOUT, CB_SSCRATCH = gr_fold.FRONT_STATS_CBS  # 20-23: the stats phase's, as gr_fold's front
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
    """The front as ONE program on gr_fold's cores and transports (the norm cores on row 2, the 6x2 down workers, the
    transport pair): stats -> u multicast -> per-stream matmul -> recip multicast -> scale and sum -> partials gather.
    Returns ``(gathered_stats, normalized', gathered_partials)`` in gr_fold's shapes and pages."""

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
    scaler_bits, scaler_dtype = gr_read.avg_scaler("chain")
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
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, p_set),
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
    # the stats transport and the core's own writer signalled (gr_fold's gate: links + 1)
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
                    [gr_read._bits(1.0), scaler_bits, gr_read._bits(gr_read.EPS)],
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
        # the device's own page 12 * rank + w (gr_fold's)
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
    # the residual once, gamma and the weight in, normalized' out, every device's stats and partial tiles landing in
    # the two gathered tensors' pages; the u row multicast to the twelve workers, the recip tiles to them, the stats /
    # partial tiles handed to the transport cores (L1); the stats' two passes, the u product, the recip's reduce and
    # rsqrt, the four-stream matmul, the scale and sum, the gate's normalized' product
    meta = fp.program_meta(
        NAME,
        "stats_recip_down_gather",
        rows,
        reads=(residual, norm_scale, down_inject),
        writes=(normalized,),
        dram_bytes=TP_SIZE * (B * T_BF16 + PT * T_FP32),
        l1_bytes=len(workers) * (fp.tensor_bytes(normalized) + B * T_FP32) + B * T_BF16 + PT * T_FP32,
        flops=6 * rows * gr_read.FLAT_WIDTH
        + 2 * rows * gr_read.FLAT_WIDTH * gr_read.PARTIAL_WIDTH
        + 2 * rows * B * gr_read.PARTIAL_WIDTH,
        cores=len(workers) + len(producers) + links,
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
    """``gr_read``'s read with this front (two programs per read: the front and ``low_rank_gate``); the chain's
    rounding knobs (scaler ``chain``, matmul ``chain`` for the gate) fixed as gr_fold fixes them."""

    return gr_read.gr_read_fused(module, residual, scaler_mode="chain", matmul="chain", read_front=read_front)


register(
    FusedKernel(
        name=NAME,
        replaces="gr_fold's front (stats, both line gathers, normalize_down as one program) with the rsqrt applied "
        "after the down projection, so the u multicast and the matmul run under the stats round trip; COMPONENT "
        "class, on by default, resolved by the GR module when gr_read is on",
        tolerance=COMPONENT,
        fused=gr_read_recip_last,
        composed=gr_fold.gr_read_fold,
        gate=None,  # the collectives need the TP4 line: the component gate is the mesh replica with the oracle (dev tools)
        component_proof="the 1x4 replica against the CPU oracle on captured residual rows (layers 2/12/47 x rows 1/5/32, the checkpoint's attention-block weights), 2026-09-27: block-input rms_rel 0.0036-0.0045 where the composed fold reads 0.0038-0.0045 (inside 1.10x + 1e-3 on every row), |bias| <= 1.9e-4, ULP p99 8-16 against the fold's 8-16; the paired 43-prompt EOS-honoured served sample +2.99 % tok/s (95 % interval +1.15..+4.84), the pass 1.2-1.7 ms shorter",
    )
)
