# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GR front v2 (``QWEN38_GR_FRONT_V2=1``, default off): ``gr_recip_last``'s front with a cheaper partials exchange.

Same cores, same compute kernels, same stats phase, same packed bytes on the wire and the same pages in the gathered
partials as the default front (bit-identical valid rows); what changes is the data movement after the partial tile exists:
  * the down workers write the two row chunks of their partial tile straight into the transport core's staging buffer
    (``raw_writer.cpp``): no scratch tile, no copy loop and write barrier on the transport core;
  * the partials phase has no barrier (parity landing slots in a persistent L1 buffer, a data semaphore per parity: gr_fold's
    ROW_MODE 4 scheme);
  * the unpack of the landed slots into the gathered pages is split over both RISCs of the transport core
    (``transport3.cpp``).
Decode rows <= 16 only; anything else runs the default front.
"""

from __future__ import annotations

import os

import ttnn

from ...contracts import TP_SIZE
from .. import gr_fold, gr_read
from .. import program as fp
from . import (
    CB_IN0,
    CB_INTERM,
    CB_NOUT,
    CB_OUT,
    CB_PACK_PARTIALS,
    CB_PACK_STATS,
    CB_PROD,
    CB_RECIP_OUT,
    CB_RECIP_W,
    CB_SOUT,
    CB_SSCALER,
    CB_SSCRATCH,
    CB_U,
    CB_W,
    CB_X2,
    CB_ZERO,
    DOWN_STREAMS,
    NAME,
    RECIP_READY,
    STATS_U_RECIP,
    U_BLOCK,
    WEIGHT_BATCH,
)
from . import stats_recip_down_gather as _default_front

ENV = "QWEN38_GR_FRONT_V2"
TRANSPORT3 = fp.kernel_source(NAME, "transport3.cpp")
RAW_WRITER = fp.kernel_source(NAME, "raw_writer.cpp")
FRONT_READER = fp.kernel_source(NAME, "front_reader.cpp")
MCAST_WRITER4 = fp.kernel_source(NAME, "mcast_writer4.cpp")
MCAST_READER3 = fp.kernel_source(NAME, "mcast_reader3.cpp")
CB_UPACK, CB_UPACK_IN = 26, 27  # the norm core's packed u chunks; the workers' landing buffer of the four cores' chunks
UNPACK_SEM = 7  # program-local, on the transport cores: BRISC -> NCRISC "the data landed, unpack your sources"
MAX_ROWS = 16
_STATE: dict = {}


def fits(rows: int) -> bool:
    """Whether the packed stats (2 of 4 tiles per link) and partials (6 of 12) of ``rows`` valid rows are one fabric packet each."""

    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    return 1 <= rows <= MAX_ROWS and 6 * 2 * rows * 64 <= payload and 2 * 2 * rows * 32 <= payload


def enabled() -> bool:
    return os.environ.get(ENV, "1") == "1"  # default on (x44 A/B -0.72 ms)


def _persistent(mesh, slot_bytes: int, cores):
    """(odd data semaphore address, landing buffer address): the second parity's data semaphore of the partials phase and
    a zeroed persistent L1 buffer on the transport cores, [call counter, 16 B][2 slot sets of TP_SIZE slots].  Created
    once per mesh and slot size, before any trace capture (a peer's packet may land whatever program this device runs).
    """

    key = (id(mesh), slot_bytes)
    if key not in _STATE:
        grid = mesh.compute_with_storage_grid_size()
        every = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        sem = ttnn.create_global_semaphore(mesh, every, 0)
        words = (16 + 2 * TP_SIZE * slot_bytes) // 4
        shard = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tensor = ttnn.zeros(
            [len(cores), words], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=memory
        )
        _STATE[key] = (sem, tensor, int(ttnn.get_global_semaphore_address(sem)), int(tensor.buffer_address()))
    return _STATE[key][2], _STATE[key][3]


def _front_reader(cores, streams, consts, runtime, gate, extra, extra_rt):
    """gr_read._reader with front_reader.cpp: ``extra`` compile args after the four accessor sets, ``extra_rt(b)`` runtime
    args after each core's constants (b = the branch the core serves)."""

    compile_args = [len(streams), *[cb for _, cb in streams], *[0] * (4 - len(streams)), len(consts)]
    for kind, cb in consts:
        compile_args += [kind, cb]
    compile_args += [0, 0] * (3 - len(consts))
    compile_args += list(gate)
    for slot in range(4):
        compile_args += fp.accessor_args(streams[min(slot, len(streams) - 1)][0])
    compile_args += list(extra)
    return fp.reader_kernel(
        FRONT_READER,
        cores,
        compile_args,
        [
            (core, [a for args in stream_args for a in args] + list(bits) + extra_rt(b))
            for b, (core, (stream_args, bits)) in enumerate(runtime)
        ],
    )


def _mcast_reader3(cores, streams, runtime, *, recv, recv2, zero_cb, extra):
    """gr_recip_last._mcast_reader2 with mcast_reader3.cpp (unpacks the packed u row): ``extra`` compile args at the end."""

    compile_args = [*recv, *recv2, len(streams), *[cb for _, cb in streams], *[0] * (2 - len(streams)), zero_cb]
    for slot in range(2):
        compile_args += fp.accessor_args(streams[min(slot, len(streams) - 1)][0])
    compile_args += list(extra)
    return fp.reader_kernel(
        MCAST_READER3,
        cores,
        compile_args,
        [(core, [a for args in stream_args for a in args]) for core, stream_args in runtime],
    )


def _transport_program(mesh, stats, partials, rows, *, semaphores, cbs, kernels, links, pack_cores):
    """gr_fold.transport_mesh_program for the two-phase front with transport3.cpp: ``stats`` is gr_fold's Transport (packed
    mode 3, barrier); ``partials`` carries the second phase's tensors, CBs and semaphores (its own protocol is transport3's).
    """

    geometry = gr_fold.line(mesh)
    cores = stats.cores[:links]
    sem_addr = gr_fold.line_semaphores(mesh)
    barrier_a, data_a = sem_addr.pair(stats.phase)
    barrier_b, data_b = sem_addr.pair(partials.phase)
    mode_a, slot_a = gr_fold._row_mode(stats, links, 3)
    if mode_a != 3:
        raise RuntimeError("the stats phase does not fit one packed packet")
    my_tiles_b = -(-partials.tiles // links)
    pack_tile = 2 * rows * 64
    slot_b = -(-(my_tiles_b * pack_tile) // 16) * 16
    if my_tiles_b * pack_tile > int(ttnn.get_tt_fabric_max_payload_size_bytes()):
        raise RuntimeError("the partials do not fit one packed packet")
    data1, land = _persistent(mesh, slot_b, cores)
    go, done = stats.semaphore_ids[0], stats.semaphore_ids[2]
    cbs = list(cbs)
    # the stats phase's landing slots + staging: one raw L1 buffer on the transport cores AND the norm cores (one address on all)
    cbs.append(fp.cb_descriptor(stats.pack_cb, gr_read.BF16, (TP_SIZE + 2) * slot_a, 1, pack_cores))
    mesh_program = ttnn.MeshProgramDescriptor()
    for rank, (coordinate, node) in enumerate(geometry.ranks()):
        base = [rank, 0, 0]
        base += [
            stats.out.buffer_address(),
            stats.local.buffer_address(),
            barrier_a,
            data_a,
            len(stats.consumers),
            stats.consumer_sem,
        ]
        base += [v for xy in stats.consumers for v in xy]
        base += [partials.out.buffer_address(), partials.local.buffer_address(), barrier_b, data_b, 0, 0]
        compile_args = []
        for i in range(links):
            args = [go, TP_SIZE, done, i, links]
            args += [
                stats.scratch_cb,
                stats.tiles,
                stats.semaphore_ids[1],
                *stats.page_strides,
                stats.rows,
                stats.pack_cb,
                slot_a,
            ]
            args += [
                partials.scratch_cb,
                partials.tiles,
                partials.semaphore_ids[1],
                *partials.page_strides,
                rows,
                slot_b,
            ]
            args += [UNPACK_SEM, data1, land]
            for t in (stats, partials):
                args += fp.accessor_args(t.out) + fp.accessor_args(t.local)
            compile_args.append(args)

        def transport(build, per_link_args):
            return [
                build(
                    TRANSPORT3,
                    ttnn.CoreRangeSet([ttnn.CoreRange(cores[i], cores[i])]),
                    compile_args[i],
                    [(cores[i], base + list(per_link_args[i]))],
                )
                for i in range(links)
            ]

        others = list(kernels(rank))
        empty = [[] for _ in range(links)]
        seed = fp.program_descriptor(
            others + transport(fp.reader_kernel, empty) + transport(fp.writer_kernel, empty),
            cbs=cbs,
            semaphores=semaphores,
        )
        backward = [
            ttnn.setup_fabric_connection(node, geometry.nodes[rank - 1], i, seed, cores[i]) if rank > 0 else []
            for i in range(links)
        ]
        forward = [
            ttnn.setup_fabric_connection(node, geometry.nodes[rank + 1], i, seed, cores[i])
            if rank + 1 < TP_SIZE
            else []
            for i in range(links)
        ]
        mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=others + transport(fp.reader_kernel, backward) + transport(fp.writer_kernel, forward),
            semaphores=list(seed.semaphores),
            cbs=list(cbs),
        )
    return mesh_program


def stats_recip_down_gather(residual, norm_scale, down_inject, *, links: int | None = None):
    """The front as ONE program on gr_fold's cores and transports (the norm cores on row 2, the 6x2 down workers, the
    transport pair): stats -> u multicast -> per-stream matmul -> recip multicast -> scale and sum -> partials gather.
    Returns ``(gathered_stats, normalized', gathered_partials)`` in gr_fold's shapes and pages."""

    rows = gr_read.residual_rows(residual)
    if not fits(rows):
        return _default_front(residual, norm_scale, down_inject, links=links)
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
        fp.cb_descriptor(CB_UPACK, gr_read.BF16, HT * 2 * rows * 32, 1, p_set),
        fp.cb_descriptor(CB_UPACK_IN, gr_read.BF16, B * HT * 2 * rows * 32, 1, pw_set),
    ]
    # the norm core's reader: the residual once (the stats and u phases share it), gamma, then the gathered stats once
    # the stats transport and the core's own writer signalled (gr_fold's gate: links + 1)
    # the norm core's reader: the residual once (the stats and u phases share it), gamma, then, once the stats transport signalled
    # and the core's own writer did (gr_fold's gate: links + 1), the gathered stats pulled from the transport core's L1
    rows_mode_slot = gr_fold._row_mode(
        gr_fold.Transport(
            gathered_stats, gathered_stats, CB_SSCRATCH, B, gr_fold.SOURCE_PRODUCERS, "stats", transports, rows=rows
        ),
        links,
        3,
    )[1]
    chunk_a = rows * 32
    stats_extra = [CB_PACK_STATS, rows_mode_slot, 2 * chunk_a, chunk_a, TP_SIZE, 512]

    def reader_for(rank):
        return _front_reader(
            p_set,
            [(residual, 0), (norm_scale, 4), (gathered_stats, 1)],
            [(gr_read.CONST_SCALER, CB_SSCALER), (gr_read.CONST_SCALER, 2), (gr_read.CONST_COL_SCALAR, 3)],
            [
                (
                    core,
                    (
                        [
                            gr_read._stream(residual, 1, HT, b * HT, 1, 0, gr_read.ROW_BATCH),
                            gr_read._stream(norm_scale, 1, HT, b * HT, 1, 0, gr_read.ROW_BATCH),
                            gr_read._stream(gathered_stats, 1, ST, b * ST, 1, 0, ST),
                        ],
                        [gr_read._bits(1.0), scaler_bits, gr_read._bits(gr_read.EPS)],
                    ),
                )
                for b, core in enumerate(producers)
            ],
            gate=(2, gr_fold.FRONT_STATS_READY, links + 1),
            extra=stats_extra,
            extra_rt=lambda b: [rank, *noc[(transports[b % links].x, transports[b % links].y)], b // links],
        )

    compute = fp.compute_kernel(
        STATS_U_RECIP,
        p_set,
        [HT, ST, U_BLOCK, CB_U, CB_SSCALER, CB_X2, CB_SOUT, CB_RECIP_OUT, CB_NOUT],
        fp32_dest=True,
    )
    receiver = _mcast_reader3(
        w_set,
        [(down_inject, CB_W)],
        [(core, [gr_read._stream(down_inject, 1, FT, w, n_tiles, 0, WEIGHT_BATCH)]) for w, core in enumerate(workers)],
        recv=(CB_IN0, FT, B, 0),
        recv2=(CB_RECIP_W, B, B, RECIP_READY),
        zero_cb=CB_ZERO,
        extra=[CB_UPACK_IN, rows * 32, 512],
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
        # b // links and the device's own page TP_SIZE * b + rank, then the core's own gate signal; B = the u row into the
        # workers' CB at tile offset b * HT; C = the recip tile into the workers' recip CB at offset b, and the
        # normalized' row written to the normalized tensor (pages b * HT ..) as the phase's extra stream
        runtime = []
        for b, core in enumerate(producers):
            x, y = noc[(transports[b % links].x, transports[b % links].y)]
            phase_a = [b // links, x, y, x, y, 0, 0, 1, 0, 0, 0, 1, 1]
            phase_b = [b * HT, *rect, 0, 0, 0, 0, 0, 0, 1, 1]
            phase_c = [b, *rect, 0, 0, 0, normalized.buffer_address(), HT, b * HT, 1, gr_read.ROW_BATCH]
            runtime.append((core, phase_a + phase_b + phase_c + list(noc[(core.x, core.y)])))
        writer3 = fp.writer_kernel(
            MCAST_WRITER4,
            p_set,
            [CB_SOUT, CB_PACK_STATS, 1, 0, gr_read.NONE_CB, s_scratch]
            + [CB_U, CB_UPACK_IN, HT, CB_UPACK, gr_read.NONE_CB, 0]
            + [CB_RECIP_OUT, CB_RECIP_W, 1, 0, CB_NOUT, RECIP_READY]
            + [gr_fold.FRONT_STATS_READY]
            + accessors
            + [rows_mode_slot, chunk_a, 512, TP_SIZE],
            runtime,
        )
        # the workers' writer: the partial tile's two row chunks into the transport core's staging buffer (the scratch CB), the
        # scratch semaphore, then the device's own page PARTIAL_TILES * rank + w
        runtime = []
        for w, core in enumerate(workers):
            x, y = noc[(transports[w % links].x, transports[w % links].y)]
            runtime.append((core, [x, y, w // links, PT * rank + w, gathered.buffer_address()]))
        writer = fp.writer_kernel(
            RAW_WRITER,
            w_set,
            [CB_OUT, gr_fold.PARTIALS_SCRATCH_CB, rows, p_scratch, *fp.accessor_args(gathered)],
            runtime,
        )
        return [reader_for(rank), compute, writer3, receiver, down, writer]

    semaphores = (
        [fp.semaphore_descriptor(0, pw_set)]
        + [fp.semaphore_descriptor(sem, wt_set) for sem in gr_fold.PARTIALS_SEMAPHORES]
        + [
            fp.semaphore_descriptor(s_scratch, ps_set),
            fp.semaphore_descriptor(gr_fold.FRONT_STATS_READY, p_set),
            fp.semaphore_descriptor(RECIP_READY, pw_set),
            fp.semaphore_descriptor(UNPACK_SEM, gr_read._core_set(list(transports))),
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
            rows=rows,
            pack_cb=CB_PACK_STATS,
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
            rows=rows,
            pack_cb=CB_PACK_PARTIALS,
        ),
    ]
    # the residual once, gamma and the weight in, normalized' out, every device's stats and partial tiles landing in
    # the two gathered tensors' pages; the u row multicast to the twelve workers, the recip tiles to them, the stats /
    # partial tiles handed to the transport cores (L1); the stats' two passes, the u product, the recip's reduce and
    # rsqrt, the four-stream matmul, the scale and sum, the gate's normalized' product
    meta = fp.program_meta(
        NAME,
        "stats_recip_down_gather_v2",
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
        _transport_program(
            mesh,
            phases[0],
            phases[1],
            rows,
            semaphores=semaphores,
            cbs=cbs,
            kernels=kernels,
            links=links,
            pack_cores=ps_set,
        ),
        meta=meta,
    )
    return gathered_stats, normalized, gathered
