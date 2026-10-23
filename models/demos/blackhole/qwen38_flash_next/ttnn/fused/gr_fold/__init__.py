# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused GR read with its collectives inside the program (fusion round 3, item 1).

The GR read of ``ttnn/fused/gr_read`` is three generic_op programs around two collectives per read.  Here the
collectives move into the program: one transport core per fabric link sends this device's tiles to the other devices
of the TP4 line over the 1D fabric (``ttnn.setup_fabric_connection`` hands a generic_op kernel a fabric connection;
the kernel is a line multicast by hop count on the 1D linear API), the tiles land in the DRAM pages the chain's
all-gather writes, behind a per-call barrier, and the compute kernels stay the proven ones.  An all-gather is data
movement, so the fold is bitwise by construction: the slot order (page ``4b + d`` for the stats, ``12d + t`` for the
partials) and every consumer's loop order are the chain's.  The design note (the fold design note, a development document that is not shipped) walks the
semaphore protocol; the three facts it stands on are pinned in the kernel and in tests/test_fused_gr_fold_static.py.

``gather_line`` = ``ttnn.all_gather(x, dim=3, cluster_axis=1)`` of a few tiles as one generic_op program per device
(``kernels/transport.cpp``), the mechanism as the smoke tool exercises it.  ``stats_gather`` (stage b) is the read's
first program with the stats all-gather inside it: the four stats cores hand their tile to the transport cores and
write it into the device's own pages; ``normalize_down_gather`` (stage c) is ``gr_read.normalize_down`` with the
partial all-gather inside it; ``stats_normalize_down_gather`` (stage d) is both as ONE program (the norm cores run
the stats body first and their reader streams the gathered stats once the stats transport signals), so the read is
two programs: it and ``low_rank_gate``.  ``gr_read_fold`` is ``gr_read``'s read with that front (registered
``gr_fold``, on by default; it needs ``gr_read`` on; ``QWEN38_FUSED_OFF=gr_fold`` keeps the merged gr_read form).  The read's rounding knobs stay the chain's: ``gr_read_fold``
hardcodes scaler and matmul mode ``chain``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import ttnn

from .. import gr_read
from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "gr_fold"
TP_SIZE = 4
TP_AXIS = 1
TRANSPORT_KERNEL = fp.kernel_source(NAME, "transport.cpp")
SEM_PROBE = fp.kernel_source(NAME, "sem_probe.cpp")
SCRATCH_CB = 0
SEM_GO, SEM_SCRATCH, SEM_DONE = (
    0,
    1,
    2,
)  # program-local, allocated first: barrier passed (BRISC -> NCRISC), producers' tiles landed, NCRISC done
SOURCE_TENSOR, SOURCE_PRODUCERS = 0, 1
PHASES = (
    "stats",
    "partials",
)  # one (barrier, data) global semaphore pair per gather phase, never cycled: the design note's proof
STATS_SCRATCH_CB = 3  # beside the stats program's CBs 0, 1, 2, 16
PARTIALS_SCRATCH_CB = 11  # beside normalize_down's CBs 0-10, 16, 17
# the transport cores per phase, one per link: the stats phase next to the four stats cores, the partials phase
# clear of normalize_down's 6x2 worker rectangle and its producers on row 2
TRANSPORT = {
    "stats": (ttnn.CoreCoord(gr_read.BRANCHES, 0), ttnn.CoreCoord(gr_read.BRANCHES + 1, 0)),
    "partials": (ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 0)),
}
TRANSPORT_CORES = TRANSPORT["stats"]
PARTIALS_SEMAPHORES = (1, 2, 3)  # normalize_down's in0 multicast owns program semaphore 0
# the stage-(d) program (stats + normalize_down as one program): the partials transport cores run both phases on one
# connection per link (kernels/transport2.cpp; a link's worker sender channel is one per direction), the stats phase
# with its own scratch semaphore; the gate is the norm cores' semaphore their reader waits on before streaming the
# gathered stats
TRANSPORT2_KERNEL = fp.kernel_source(NAME, "transport2.cpp")
FRONT_STATS_SCRATCH_SEM = 4  # on the transport cores, beside the partials' (go, scratch, done) = 1, 2, 3
FRONT_STATS_READY = 5  # on the norm cores, beside the in0 multicast's 0
FRONT_STATS_CBS = (
    20,
    21,
    22,
    23,
)  # the stats phase's scaler (fp32), x^2 (fp32), stats tile (bf16), transport scratch (bf16)
STATS_NORM = fp.kernel_source(NAME, "stats_norm_compute.cpp")
MCAST_WRITER2 = fp.kernel_source(NAME, "mcast_writer2.cpp")
PROBED_CORES = TRANSPORT["stats"] + TRANSPORT["partials"]
_SEMAPHORES: dict[int, "LineSemaphores"] = {}
_LINES: dict[int, "Line"] = {}


class LineSemaphores:
    """The global semaphores of the line transport: a (barrier, data) pair per gather phase, one address on every
    device, created once per mesh before any trace capture."""

    def __init__(self, mesh) -> None:
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.handles = {
            phase: (ttnn.create_global_semaphore(mesh, cores, 0), ttnn.create_global_semaphore(mesh, cores, 0))
            for phase in PHASES
        }
        self.addresses = {
            phase: (int(ttnn.get_global_semaphore_address(barrier)), int(ttnn.get_global_semaphore_address(data)))
            for phase, (barrier, data) in self.handles.items()
        }

    def pair(self, phase: str) -> tuple[int, int]:
        return self.addresses[phase]


def line_semaphores(mesh) -> LineSemaphores:
    key = id(mesh)
    if key not in _SEMAPHORES:
        _SEMAPHORES[key] = LineSemaphores(mesh)
    return _SEMAPHORES[key]


class Line:
    """The TP line as the fabric sees it: rank r -> (mesh coordinate, fabric node id), checked once per mesh: the
    mesh is the model's ``(1, 4)`` with the line along axis 1, every neighbour pair is one hop in a direction
    consistent along the line (forward one way, backward the other), the degree histogram is {1: 2, 2: 2}, and the
    link indices usable towards each neighbour are recorded (``links`` = the count every pair offers)."""

    def __init__(self, mesh) -> None:
        shape = tuple(int(v) for v in mesh.shape)
        if shape != (1, TP_SIZE):
            raise ValueError(f"the line transport needs the (1, {TP_SIZE}) mesh, got {shape}")
        self.coordinates = [ttnn.MeshCoordinate(0, rank) for rank in range(TP_SIZE)]
        self.nodes = [mesh.get_fabric_node_id(coordinate) for coordinate in self.coordinates]
        forward = [ttnn.get_eth_forwarding_direction(self.nodes[r], self.nodes[r + 1]) for r in range(TP_SIZE - 1)]
        backward = [ttnn.get_eth_forwarding_direction(self.nodes[r + 1], self.nodes[r]) for r in range(TP_SIZE - 1)]
        if (
            None in forward
            or None in backward
            or len(set(forward)) != 1
            or len(set(backward)) != 1
            or forward[0] == backward[0]
        ):
            raise RuntimeError(f"mesh axis {TP_AXIS} is not the fabric's line: forward {forward} backward {backward}")
        degree = Counter()
        for rank in range(TP_SIZE):
            for other in range(TP_SIZE):
                if other != rank and ttnn.get_eth_forwarding_direction(self.nodes[rank], self.nodes[other]) is not None:
                    degree[rank] += 1
        # every rank can forward to every other on a line; the immediate-neighbour count is what the transport uses
        neighbours = {rank: (rank > 0) + (rank + 1 < TP_SIZE) for rank in range(TP_SIZE)}
        if Counter(neighbours.values()) != Counter({1: 2, 2: 2}) or any(
            degree[r] != TP_SIZE - 1 for r in range(TP_SIZE)
        ):
            raise RuntimeError(
                f"line degree histogram is not {{1: 2, 2: 2}}: neighbours {neighbours} reachable {dict(degree)}"
            )
        link_counts = {
            len(ttnn.get_forwarding_link_indices(self.nodes[a], self.nodes[b]))
            for a, b in [(r, r + 1) for r in range(TP_SIZE - 1)] + [(r + 1, r) for r in range(TP_SIZE - 1)]
        }
        if len(link_counts) != 1 or min(link_counts) < 1:
            raise RuntimeError(f"the line's neighbour pairs offer different link counts: {link_counts}")
        self.links = link_counts.pop()
        self.forward_direction, self.backward_direction = forward[0], backward[0]

    def ranks(self):
        return list(zip(self.coordinates, self.nodes))


def line(mesh) -> Line:
    key = id(mesh)
    if key not in _LINES:
        _LINES[key] = Line(mesh)
    return _LINES[key]


def _tiles(tensor) -> int:
    padded = tuple(tensor.padded_shape)
    return padded[0] * padded[1] * (padded[2] // fp.TILE) * (padded[3] // fp.TILE)


@dataclass(frozen=True)
class Transport:
    """One gather phase inside a program: tiles of ``local`` (SOURCE_TENSOR) or of producer cores (SOURCE_PRODUCERS)
    land in ``out`` at page ``t * page_strides[0] + rank * page_strides[1]``, moved by ``cores[:links]`` (one per
    link, tiles dealt round robin) with the program-local semaphore ids (go, scratch, done); the transport cores
    raise ``consumer_sem`` on each of ``consumers`` (NoC x, y) once every peer's tile has landed; ``delays`` = {rank:
    (before_arrive, after_reset)} spin cycles for the skew soak (the first transport's, in a two-phase program)."""

    out: Any
    local: Any
    scratch_cb: int
    tiles: int
    source: int
    phase: str
    cores: tuple
    semaphore_ids: tuple = (SEM_GO, SEM_SCRATCH, SEM_DONE)
    page_strides: tuple = (TP_SIZE, 1)
    consumers: tuple = ()
    consumer_sem: int = 0
    delays: Any = None


def transport_mesh_program(
    mesh, transports, *, semaphores, cbs, kernels=lambda rank: [], links: int = 1
) -> "ttnn.MeshProgramDescriptor":
    """One ProgramDescriptor per mesh coordinate: ``kernels(rank)`` plus, per link, the transport core's reader (NCRISC,
    backward) and writer (BRISC, forward) with the device's rank and fabric connections.  One ``Transport`` runs
    ``kernels/transport.cpp``; two run ``kernels/transport2.cpp`` on the SAME cores, both phases on one connection per
    RISC (a fabric link's worker sender channel is one per direction, so a program holds at most one open sender per
    link per direction: the design note's link budget); they share ``go`` and ``done`` and have distinct scratch
    semaphores and phases.  The program's own semaphores (``semaphores``, fixed ids) are in the descriptor before
    ``setup_fabric_connection`` adds a connection's two, always in the order backward then forward, so the ids are the
    same on every build (ends: 2 connection semaphores, middles: 4); ``setup_fabric_connection`` mutates the
    descriptor it is given and returns the kernel's runtime args, so the transport kernels are built twice: once to
    seed the descriptor, once with the args."""

    geometry = line(mesh)
    if not 1 <= len(transports) <= 2:
        raise ValueError(f"a program carries one or two transports, got {len(transports)}")
    first = transports[0]
    cores = first.cores[:links]
    for t in transports:
        if not 1 <= links <= min(len(t.cores), geometry.links, t.tiles):
            raise ValueError(
                f"links must be 1..{min(len(t.cores), geometry.links, t.tiles)} for the {t.phase} transport, got {links}"
            )
        if t.cores[:links] != cores:
            raise ValueError(
                "the transports of one program run on the same cores (one open sender per link per direction)"
            )
        if (t.semaphore_ids[0], t.semaphore_ids[2]) != (first.semaphore_ids[0], first.semaphore_ids[2]):
            raise ValueError("the transports of one program share the go and done semaphores")
    if len({t.phase for t in transports}) != len(transports) or len({t.semaphore_ids[1] for t in transports}) != len(
        transports
    ):
        raise ValueError("one transport per phase per program, each with its own scratch semaphore")
    addresses = {t.phase: line_semaphores(mesh).pair(t.phase) for t in transports}
    go, done = first.semaphore_ids[0], first.semaphore_ids[2]
    mesh_program = ttnn.MeshProgramDescriptor()
    for rank, (coordinate, node) in enumerate(geometry.ranks()):
        before, after = (first.delays or {}).get(rank, (0, 0))
        base = [rank, before, after]
        for t in transports:  # the phase block: 6 words + the consumers' NoC (x, y)
            barrier, data = addresses[t.phase]
            base += [t.out.buffer_address(), t.local.buffer_address(), barrier, data, len(t.consumers), t.consumer_sem]
            base += [v for xy in t.consumers for v in xy]
        compile_args = []
        for i in range(links):
            if len(transports) == 1:
                args = [
                    first.scratch_cb,
                    first.tiles,
                    go,
                    TP_SIZE,
                    first.source,
                    first.semaphore_ids[1],
                    done,
                    i,
                    links,
                    *first.page_strides,
                ]
            else:
                args = [go, TP_SIZE, done, i, links]
                for t in transports:
                    args += [t.scratch_cb, t.tiles, t.source, t.semaphore_ids[1], *t.page_strides]
            for t in transports:
                args += fp.accessor_args(t.out) + fp.accessor_args(t.local)
            compile_args.append(args)
        kernel = TRANSPORT_KERNEL if len(transports) == 1 else TRANSPORT2_KERNEL

        def transport(build, per_link_args):
            return [
                build(
                    kernel,
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
            (
                ttnn.setup_fabric_connection(node, geometry.nodes[rank + 1], i, seed, cores[i])
                if rank + 1 < TP_SIZE
                else []
            )
            for i in range(links)
        ]
        mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=others + transport(fp.reader_kernel, backward) + transport(fp.writer_kernel, forward),
            semaphores=list(seed.semaphores),
            cbs=list(cbs),
        )
    return mesh_program


def gather_line(local, *, links: int = 1, delays=None, dim: int = 3):
    """The chain's all-gather of a one-row-block TILE tensor as one generic_op program per device, moved by the
    transport cores over the fabric.  ``dim=3``: ``ttnn.all_gather(local, dim=3, cluster_axis=1)`` of a one-tile-wide
    ``[1, B, rows, 32]`` -> ``[1, B, rows, 128]``, page ``t * 4 + rank`` (the stats phase); ``dim=0``:
    ``all_gather_async(local, dim=0)`` of ``[1, 1, rows, W]`` -> ``[4, 1, rows, W]``, page ``rank * T + t`` (the
    partials phase)."""

    shape = tuple(local.shape)
    if len(shape) != 4 or local.layout != ttnn.TILE_LAYOUT or local.padded_shape[2] != fp.TILE:
        raise ValueError(f"gather_line takes a one-row-block TILE tensor, got {shape} {local.layout}")
    if dim == 3 and local.padded_shape[-1] != fp.TILE:
        raise ValueError(f"gather_line(dim=3) takes a tensor one tile wide, got {shape}")
    if dim == 0 and shape[:2] != (1, 1):
        raise ValueError(f"gather_line(dim=0) takes [1, 1, rows, W], got {shape}")
    mesh = local.device()
    tiles = _tiles(local)
    tile_bytes = fp.TILE_BYTES[local.dtype]
    max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    if tile_bytes > max_payload:
        raise ValueError(f"a {tile_bytes}-byte tile exceeds the fabric payload of {max_payload} bytes")
    if dim == 3:
        out = fp.allocate((shape[0], shape[1], shape[2], shape[3] * TP_SIZE), local.dtype, ttnn.TILE_LAYOUT, mesh)
        phase, strides = "stats", (TP_SIZE, 1)
    else:
        out = fp.allocate((TP_SIZE, 1, shape[2], shape[3]), local.dtype, ttnn.TILE_LAYOUT, mesh)
        phase, strides = "partials", (1, tiles)
    transports = TRANSPORT[phase]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports[:links]])
    cbs = [fp.cb_descriptor(SCRATCH_CB, local.dtype, tile_bytes, tiles, cores)]
    semaphores = [fp.semaphore_descriptor(sem, cores) for sem in (SEM_GO, SEM_SCRATCH, SEM_DONE)]
    spec = Transport(
        out, local, SCRATCH_CB, tiles, SOURCE_TENSOR, phase, transports, page_strides=strides, delays=delays
    )
    # the local tiles in, every device's tiles landing in the gathered pages (the own copy and the peers' over the
    # fabric); the tiles staged in the transport cores' scratch (L1); data movement
    meta = fp.program_meta(
        NAME,
        "gather_line",
        shape[2],
        reads=(local,),
        dram_bytes=TP_SIZE * tiles * tile_bytes,
        l1_bytes=tiles * tile_bytes,
        cores=links,
        outputs=((out, None),),  # every device holds every device's tiles
    )
    fp.run_program(
        [local, out], transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, links=links), meta=meta
    )
    return fp.stamp_topology(out, local)


def read_semaphores(mesh):
    """Every device's line-transport global semaphores read back from every phase's transport cores: {phase:
    {"barrier": [per probed core [per device value]], "data": ...}}; all zero between calls."""

    addresses = [a for phase in PHASES for a in line_semaphores(mesh).pair(phase)]
    probed = list(PROBED_CORES)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in probed])
    out = fp.allocate((1, 1, fp.TILE, fp.TILE * len(probed)), ttnn.uint32, ttnn.TILE_LAYOUT, mesh)
    probe = fp.reader_kernel(
        SEM_PROBE,
        cores,
        [len(addresses)] + fp.accessor_args(out),
        [(core, [out.buffer_address(), slot] + addresses) for slot, core in enumerate(probed)],
    )
    fp.run_program(
        [out, out],
        fp.program_descriptor([probe], cbs=[fp.cb_descriptor(0, ttnn.uint32, fp.TILE_BYTES[ttnn.uint32], 1, cores)]),
        meta=fp.program_meta(NAME, "sem_probe", 1, writes=(out,), cores=len(probed)),
    )
    values = []
    for shard in ttnn.get_device_tensors(out):
        words = ttnn.to_torch(shard).reshape(fp.TILE, len(probed), fp.TILE)[0]
        values.append([[int(words[slot, i]) for i in range(len(addresses))] for slot in range(len(probed))])
    ttnn.deallocate(out)
    result = {}
    for p, phase in enumerate(PHASES):
        result[phase] = {
            name: [[values[device][slot][2 * p + k] for device in range(len(values))] for slot in range(len(probed))]
            for k, name in enumerate(("barrier", "data"))
        }
    return result


def stats_gather(residual, *, links: int | None = None):
    """The read's first program with the stats all-gather inside it: stats core b (the chain's rmsnorm_pre_allgather
    arithmetic, ``gr_read.stats``) writes its bf16 tile into page ``4b + rank`` of the gathered ``[1, 4, rows, 128]``
    and into transport core ``b % links``'s scratch slot ``b // links``, raising that core's scratch semaphore; the
    transport cores (one per link the line offers, by default) send the tiles to the other devices' pages.
    Replaces ``gr_read.stats`` + ``ttnn.all_gather``: the same tiles in the same pages."""

    rows = gr_read.residual_rows(residual)
    mesh = residual.device()
    transports = TRANSPORT["stats"]
    links = min(len(transports), line(mesh).links) if links is None else links  # every link the line offers
    transports = transports[:links]
    out = fp.allocate((1, gr_read.BRANCHES, rows, fp.TILE * TP_SIZE), gr_read.BF16, ttnn.TILE_LAYOUT, mesh)
    work = fp.split_work(gr_read.BRANCHES, mesh)
    if any(w.core in transports for w in work):
        raise RuntimeError(f"a transport core {transports} is one of the stats cores")
    stats_cores = fp.core_set(work)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in [w.core for w in work] + list(transports)])
    noc = gr_read.noc_map(mesh)
    cbs = [
        fp.cb_descriptor(0, gr_read.BF16, gr_read.TILE_BF16, gr_read.HIDDEN_TILES, stats_cores),
        fp.cb_descriptor(1, gr_read.FP32, gr_read.TILE_FP32, 1, stats_cores),
        fp.cb_descriptor(2, gr_read.FP32, gr_read.TILE_FP32, gr_read.HIDDEN_TILES, stats_cores),
        fp.cb_descriptor(16, gr_read.BF16, gr_read.TILE_BF16, 1, stats_cores),
        fp.cb_descriptor(STATS_SCRATCH_CB, gr_read.BF16, gr_read.TILE_BF16, gr_read.BRANCHES, all_cores),
    ]
    semaphores = [fp.semaphore_descriptor(sem, all_cores) for sem in (SEM_GO, SEM_SCRATCH, SEM_DONE)]
    reader = gr_read._reader(
        stats_cores,
        [(residual, 0)],
        [(gr_read.CONST_SCALER, 1)],
        [
            (
                w.core,
                (
                    [gr_read._stream(residual, 1, gr_read.HIDDEN_TILES, w.start * gr_read.HIDDEN_TILES, 1, 0, 4)],
                    [gr_read._bits(1.0)],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(gr_read.STATS, stats_cores, [gr_read.HIDDEN_TILES], fp32_dest=True)

    def kernels(rank):
        # the stats writer of device `rank`: the tile into transport core (b % links)'s scratch slot b // links (a 1x1
        # multicast + that core's scratch semaphore) and into the device's own page 4b + rank
        runtime = []
        for w in work:
            x, y = noc[(transports[w.start % links].x, transports[w.start % links].y)]
            runtime.append((w.core, (w.start // links, (x, y, x, y), (w.start * TP_SIZE + rank, 1), (0, 0, 1, 1))))
        writer = gr_read._mcast_writer(
            stats_cores, runtime, src_cb=16, dst_cb=STATS_SCRATCH_CB, tiles=1, tiles_tensor=out, sem=SEM_SCRATCH
        )
        return [reader, compute, writer]

    spec = Transport(out, out, STATS_SCRATCH_CB, gr_read.BRANCHES, SOURCE_PRODUCERS, "stats", TRANSPORT["stats"])
    # the residual in, every device's stats tiles landing in the gathered pages; the four tiles handed to the
    # transport cores (L1); square and sum per element
    meta = fp.program_meta(
        NAME,
        "stats_gather",
        rows,
        reads=(residual,),
        dram_bytes=TP_SIZE * gr_read.BRANCHES * gr_read.TILE_BF16,
        l1_bytes=gr_read.BRANCHES * gr_read.TILE_BF16,
        flops=2 * rows * gr_read.FLAT_WIDTH,
        cores=len(work) + links,
        outputs=((out, None),),
    )
    fp.run_program(
        [residual, out],
        transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=kernels, links=links),
        meta=meta,
    )
    return fp.stamp_topology(out, residual)


def normalize_down_gather(residual, gathered_stats, norm_scale, down_inject, *, links: int | None = None):
    """``gr_read.normalize_down`` with the partial all-gather inside it (stage c): the same four norm cores, in0
    multicast and twelve down workers, but worker w writes its fp32 partial tile into page ``12 * rank + w`` of the
    gathered ``[4, 1, rows, 384]`` (the page order of the chain's ``all_gather_async(dim=0)``) and into transport core
    ``w % links``'s scratch slot ``w // links``; the transport cores send the twelve tiles to the other devices' pages.
    Returns ``(normalized, gathered_partials)``; ``low_rank_gate`` reads the gathered tensor exactly as today."""

    rows = gr_read.residual_rows(residual)
    gr_read._expect(
        gathered_stats, (1, gr_read.BRANCHES, rows, fp.TILE * gr_read.STATS_TILES), gr_read.BF16, "GR gathered stats"
    )
    gr_read._expect(
        norm_scale, (1, gr_read.BRANCHES, fp.TILE, gr_read.LOCAL_HIDDEN), gr_read.FP32, "GR norm_scale rows"
    )
    gr_read._expect(down_inject, (1, 1, gr_read.FLAT_WIDTH, gr_read.PARTIAL_WIDTH), gr_read.BF16, "GR down_inject")
    mesh = residual.device()
    transports = TRANSPORT["partials"]
    links = min(len(transports), line(mesh).links) if links is None else links
    transports = transports[:links]
    normalized = fp.allocate((1, 1, rows, gr_read.FLAT_WIDTH), gr_read.BF16, ttnn.TILE_LAYOUT, mesh)
    gathered = fp.allocate((TP_SIZE, 1, rows, gr_read.PARTIAL_WIDTH), gr_read.FP32, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = gr_read.avg_scaler("chain")
    n_tiles = gr_read._tiles_wide(down_inject)
    workers, producers, rect = gr_read._rectangle(mesh, 6, 2, rows_above=gr_read.BRANCHES)
    if any(c in transports for c in workers + producers):
        raise RuntimeError(f"the partials transport cores {transports} overlap normalize_down's cores")
    noc = gr_read.noc_map(mesh)
    w_set, p_set = gr_read._core_set(workers), gr_read._core_set(producers)
    all_set = gr_read._core_set(workers + producers)
    wt_set = gr_read._core_set(workers + list(transports))
    T_BF16, T_FP32, HT, ST, FT, PT = (
        gr_read.TILE_BF16,
        gr_read.TILE_FP32,
        gr_read.HIDDEN_TILES,
        gr_read.STATS_TILES,
        gr_read.FLAT_TILES,
        gr_read.PARTIAL_TILES,
    )
    cbs = [
        fp.cb_descriptor(0, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(1, gr_read.BF16, T_BF16, ST, p_set),
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, p_set),
        fp.cb_descriptor(3, gr_read.BF16, T_BF16, 1, p_set),
        fp.cb_descriptor(4, gr_read.FP32, T_FP32, HT, p_set),
        fp.cb_descriptor(5, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(6, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(7, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(16, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(8, gr_read.BF16, T_BF16, FT, all_set),
        fp.cb_descriptor(9, gr_read.BF16, T_BF16, FT, w_set),
        fp.cb_descriptor(10, gr_read.FP32, T_FP32, 1, w_set),
        fp.cb_descriptor(17, gr_read.FP32, T_FP32, 1, w_set),
        fp.cb_descriptor(PARTIALS_SCRATCH_CB, gr_read.FP32, T_FP32, PT, wt_set),
    ]
    reader = gr_read._reader(
        p_set,
        [(gathered_stats, 1), (residual, 0), (norm_scale, 4)],
        [(gr_read.CONST_SCALER, 2), (gr_read.CONST_COL_SCALAR, 3)],
        [
            (
                core,
                (
                    [
                        gr_read._stream(gathered_stats, 1, ST, b * ST, 1, 0, ST),
                        gr_read._stream(residual, 1, HT, b * HT, 1, 0, 4),
                        gr_read._stream(norm_scale, 1, HT, b * HT, 1, 0, 4),
                    ],
                    [scaler_bits, gr_read._bits(gr_read.EPS)],
                ),
            )
            for b, core in enumerate(producers)
        ],
    )
    norm = fp.compute_kernel(gr_read.NORM, p_set, [HT, ST, 4, 16], fp32_dest=True, unpack_to_dest_fp32=(4, 7))
    sender = gr_read._mcast_writer(
        p_set,
        [(core, (b * HT, rect, (b * HT, 1), (0, 0, 1, 1))) for b, core in enumerate(producers)],
        src_cb=16,
        dst_cb=8,
        tiles=HT,
        tiles_tensor=normalized,
    )
    receiver = gr_read._mcast_reader(
        w_set,
        [(down_inject, 9)],
        [(core, [gr_read._stream(down_inject, 1, FT, w, n_tiles, 0, 8)]) for w, core in enumerate(workers)],
        recv_cb=8,
        recv_tiles=FT,
        senders=gr_read.BRANCHES,
    )
    down = fp.compute_kernel(gr_read.DOWN, w_set, [FT, 8, 8, 9, 17, gr_read.DOWN_SPILL, 10], fp32_dest=True)
    go, scratch, done = PARTIALS_SEMAPHORES

    def kernels(rank):
        # worker w of device `rank`: its partial tile into transport core (w % links)'s scratch slot w // links (a 1x1
        # multicast + that core's scratch semaphore) and into the device's own page 12 * rank + w
        runtime = []
        for w, core in enumerate(workers):
            x, y = noc[(transports[w % links].x, transports[w % links].y)]
            runtime.append((core, (w // links, (x, y, x, y), (PT * rank + w, 1), (0, 0, 1, 1))))
        writer = gr_read._mcast_writer(
            w_set, runtime, src_cb=17, dst_cb=PARTIALS_SCRATCH_CB, tiles=1, tiles_tensor=gathered, sem=scratch
        )
        return [reader, norm, sender, receiver, down, writer]

    semaphores = [fp.semaphore_descriptor(0, all_set)] + [
        fp.semaphore_descriptor(sem, wt_set) for sem in (go, scratch, done)
    ]
    spec = Transport(
        gathered,
        gathered,
        PARTIALS_SCRATCH_CB,
        PT,
        SOURCE_PRODUCERS,
        "partials",
        transports,
        semaphore_ids=PARTIALS_SEMAPHORES,
        page_strides=(1, PT),
    )
    # gr_read.normalize_down's operands in and its flat row out, every device's fp32 partial tiles landing in the
    # gathered pages; the flat row multicast into the twelve down cores' CB and the partial tiles handed to the
    # transport cores (L1); the norm's four passes, then the down+inject matmul
    meta = fp.program_meta(
        NAME,
        "normalize_down_gather",
        rows,
        reads=(residual, gathered_stats, norm_scale, down_inject),
        writes=(normalized,),
        dram_bytes=TP_SIZE * PT * T_FP32,
        l1_bytes=len(workers) * fp.tensor_bytes(normalized) + PT * T_FP32,
        flops=4 * rows * gr_read.FLAT_WIDTH + 2 * rows * gr_read.FLAT_WIDTH * gr_read.PARTIAL_WIDTH,
        cores=len(workers) + len(producers) + links,
        outputs=((normalized, 3), (gathered, None)),
    )
    fp.run_program(
        [residual, gathered_stats, norm_scale, down_inject, normalized, gathered],
        transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=kernels, links=links),
        meta=meta,
    )
    return normalized, gathered


def stats_normalize_down_gather(residual, norm_scale, down_inject, *, links: int | None = None):
    """Stage (d): ``stats_gather`` and ``normalize_down_gather`` as ONE program.  Norm core b runs stats_compute's
    body then norm_compute's (``kernels/stats_norm_compute.cpp``); its two-phase writer hands the stats tile to the
    transport core ``b % links`` and writes the device's own page, then multicasts the normalized row to the twelve
    down workers; the transport cores (the partials pair (6,0)/(7,0)) run both gather phases on one connection per
    link (``kernels/transport2.cpp``: the stats phase, then the partials phase).  The norm cores' reader streams the
    gathered stats pages only once its gate semaphore counts ``links + 1``: one from each transport core after the
    stats phase's data wait (every peer's tiles landed) and one from the core's own writer after its phase A (its
    own page written; the transport's scratch signal precedes that write, so nothing else orders it inside one
    program).  The workers are normalize_down_gather's.  Returns ``(gathered_stats, normalized, gathered_partials)``."""

    rows = gr_read.residual_rows(residual)
    gr_read._expect(
        norm_scale, (1, gr_read.BRANCHES, fp.TILE, gr_read.LOCAL_HIDDEN), gr_read.FP32, "GR norm_scale rows"
    )
    gr_read._expect(down_inject, (1, 1, gr_read.FLAT_WIDTH, gr_read.PARTIAL_WIDTH), gr_read.BF16, "GR down_inject")
    mesh = residual.device()
    transports = TRANSPORT["partials"]
    links = min(len(transports), line(mesh).links) if links is None else links
    transports = transports[:links]
    gathered_stats = fp.allocate((1, gr_read.BRANCHES, rows, fp.TILE * TP_SIZE), gr_read.BF16, ttnn.TILE_LAYOUT, mesh)
    normalized = fp.allocate((1, 1, rows, gr_read.FLAT_WIDTH), gr_read.BF16, ttnn.TILE_LAYOUT, mesh)
    gathered = fp.allocate((TP_SIZE, 1, rows, gr_read.PARTIAL_WIDTH), gr_read.FP32, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = gr_read.avg_scaler("chain")
    n_tiles = gr_read._tiles_wide(down_inject)
    workers, producers, rect = gr_read._rectangle(mesh, 6, 2, rows_above=gr_read.BRANCHES)
    if any(c in transports for c in workers + producers):
        raise RuntimeError(f"the transport cores {transports} overlap normalize_down's cores")
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
    c_sscaler, c_x2, c_sout, c_sscratch = FRONT_STATS_CBS
    cbs = [
        fp.cb_descriptor(0, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(1, gr_read.BF16, T_BF16, ST, p_set),
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, p_set),
        fp.cb_descriptor(3, gr_read.BF16, T_BF16, 1, p_set),
        fp.cb_descriptor(4, gr_read.FP32, T_FP32, HT, p_set),
        fp.cb_descriptor(5, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(6, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(7, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(16, gr_read.BF16, T_BF16, HT, p_set),
        fp.cb_descriptor(c_sscaler, gr_read.FP32, T_FP32, 1, p_set),
        fp.cb_descriptor(c_x2, gr_read.FP32, T_FP32, HT, p_set),
        fp.cb_descriptor(c_sout, gr_read.BF16, T_BF16, 1, p_set),
        fp.cb_descriptor(c_sscratch, gr_read.BF16, T_BF16, gr_read.BRANCHES, ps_set),
        fp.cb_descriptor(8, gr_read.BF16, T_BF16, FT, pw_set),
        fp.cb_descriptor(9, gr_read.BF16, T_BF16, FT, w_set),
        fp.cb_descriptor(10, gr_read.FP32, T_FP32, 1, w_set),
        fp.cb_descriptor(17, gr_read.FP32, T_FP32, 1, w_set),
        fp.cb_descriptor(PARTIALS_SCRATCH_CB, gr_read.FP32, T_FP32, PT, wt_set),
    ]
    # the norm core's reader: the residual for the stats phase, gamma, the residual again for the norm phase (its CB
    # frees when the stats phase pops), then the gathered stats once the stats transport signalled every peer's tile
    reader = gr_read._reader(
        p_set,
        [(residual, 0), (norm_scale, 4), (residual, 0), (gathered_stats, 1)],
        [(gr_read.CONST_SCALER, c_sscaler), (gr_read.CONST_SCALER, 2), (gr_read.CONST_COL_SCALAR, 3)],
        [
            (
                core,
                (
                    [
                        gr_read._stream(residual, 1, HT, b * HT, 1, 0, 4),
                        gr_read._stream(norm_scale, 1, HT, b * HT, 1, 0, 4),
                        gr_read._stream(residual, 1, HT, b * HT, 1, 0, 4),
                        gr_read._stream(gathered_stats, 1, ST, b * ST, 1, 0, ST),
                    ],
                    [gr_read._bits(1.0), scaler_bits, gr_read._bits(gr_read.EPS)],
                ),
            )
            for b, core in enumerate(producers)
        ],
        gate=(3, FRONT_STATS_READY, links + 1),
    )
    compute = fp.compute_kernel(
        STATS_NORM, p_set, [HT, ST, 4, 16, c_sscaler, c_x2, c_sout], fp32_dest=True, unpack_to_dest_fp32=(4, 7)
    )
    receiver = gr_read._mcast_reader(
        w_set,
        [(down_inject, 9)],
        [(core, [gr_read._stream(down_inject, 1, FT, w, n_tiles, 0, 8)]) for w, core in enumerate(workers)],
        recv_cb=8,
        recv_tiles=FT,
        senders=gr_read.BRANCHES,
    )
    down = fp.compute_kernel(gr_read.DOWN, w_set, [FT, 8, 8, 9, 17, gr_read.DOWN_SPILL, 10], fp32_dest=True)
    p_go, p_scratch, p_done = PARTIALS_SEMAPHORES
    s_scratch = FRONT_STATS_SCRATCH_SEM
    accessors = (
        fp.accessor_args(gathered_stats)
        + fp.accessor_args(gathered_stats)
        + fp.accessor_args(normalized)
        + fp.accessor_args(normalized)
    )

    def kernels(rank):
        # the norm cores' two-phase writer: A = the stats tile into transport core (b % links)'s stats scratch slot
        # b // links and the device's own page 4b + rank, then the core's own gate signal; B = the normalized row into
        # the workers' CB 8 and the normalized tensor
        runtime = []
        for b, core in enumerate(producers):
            x, y = noc[(transports[b % links].x, transports[b % links].y)]
            phase_a = [b // links, x, y, x, y, gathered_stats.buffer_address(), b * TP_SIZE + rank, 1, 0, 0, 0, 1, 1]
            phase_b = [b * HT, *rect, normalized.buffer_address(), b * HT, 1, 0, 0, 0, 1, 1]
            runtime.append((core, phase_a + phase_b + list(noc[(core.x, core.y)])))
        writer2 = fp.writer_kernel(
            MCAST_WRITER2,
            p_set,
            [c_sout, c_sscratch, 1, 1, gr_read.NONE_CB, s_scratch, 16, 8, HT, 1, gr_read.NONE_CB, 0, FRONT_STATS_READY]
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
            w_set, runtime, src_cb=17, dst_cb=PARTIALS_SCRATCH_CB, tiles=1, tiles_tensor=gathered, sem=p_scratch
        )
        return [reader, compute, writer2, receiver, down, writer]

    semaphores = (
        [fp.semaphore_descriptor(0, pw_set)]
        + [fp.semaphore_descriptor(sem, wt_set) for sem in PARTIALS_SEMAPHORES]
        + [fp.semaphore_descriptor(s_scratch, ps_set), fp.semaphore_descriptor(FRONT_STATS_READY, p_set)]
    )
    phases = [
        Transport(
            gathered_stats,
            gathered_stats,
            c_sscratch,
            gr_read.BRANCHES,
            SOURCE_PRODUCERS,
            "stats",
            transports,
            semaphore_ids=(p_go, s_scratch, p_done),
            consumers=tuple(noc[(c.x, c.y)] for c in producers),
            consumer_sem=FRONT_STATS_READY,
        ),
        Transport(
            gathered,
            gathered,
            PARTIALS_SCRATCH_CB,
            PT,
            SOURCE_PRODUCERS,
            "partials",
            transports,
            semaphore_ids=PARTIALS_SEMAPHORES,
            page_strides=(1, PT),
        ),
    ]
    # the residual twice (the stats phase, then the norm phase), gamma and the weight in, the flat row out, every
    # device's stats and partial tiles landing in the two gathered tensors' pages; the flat row multicast to the
    # twelve down cores and the stats / partial tiles handed to the transport cores (L1); the stats' two passes, the
    # norm's four, the down+inject matmul
    meta = fp.program_meta(
        NAME,
        "stats_normalize_down_gather",
        rows,
        reads=(residual, residual, norm_scale, down_inject),
        writes=(normalized,),
        dram_bytes=TP_SIZE * (gr_read.BRANCHES * T_BF16 + PT * T_FP32),
        l1_bytes=len(workers) * fp.tensor_bytes(normalized) + gr_read.BRANCHES * T_BF16 + PT * T_FP32,
        flops=6 * rows * gr_read.FLAT_WIDTH + 2 * rows * gr_read.FLAT_WIDTH * gr_read.PARTIAL_WIDTH,
        cores=len(workers) + len(producers) + links,
        outputs=((gathered_stats, None), (normalized, 3), (gathered, None)),
    )
    fp.run_program(
        [residual, norm_scale, down_inject, gathered_stats, normalized, gathered],
        transport_mesh_program(mesh, phases, semaphores=semaphores, cbs=cbs, kernels=kernels, links=links),
        meta=meta,
    )
    return gathered_stats, normalized, gathered


def read_front(residual, gamma_rows, down_inject):
    """``gr_read_fused``'s ``read_front`` hook: the stage-(d) program; the gathered stats stay internal."""

    gathered_stats, normalized, gathered_partials = stats_normalize_down_gather(residual, gamma_rows, down_inject)
    ttnn.deallocate(gathered_stats)
    return normalized, gathered_partials


def gr_read_fold(module, residual):
    """``gr_read``'s read with both all-gathers and the stats program folded into normalize_down (stage d: the
    read is two programs, ``stats_normalize_down_gather`` and ``low_rank_gate``); the chain's rounding knobs (scaler
    ``chain``, matmul ``chain``) are fixed here, not read from the re-pin environment switches."""

    return gr_read.gr_read_fused(module, residual, scaler_mode="chain", matmul="chain", read_front=read_front)


register(
    FusedKernel(
        name=NAME,
        replaces="gr_read's stats, all_gather, normalize_down and all_gather_async (4 of the read's 5 programs) as ONE "
        "program with the collectives inside it; on by default, resolved by the GR module when gr_read is on",
        tolerance=BITWISE,
        fused=gr_read_fold,
        composed=gr_read.gr_read_fused,
        gate=None,  # the collective needs the TP4 line: the component gate is the mesh replica (dev tools)
    )
)
