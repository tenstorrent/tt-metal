# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The TP line transport library: generic_op programs that move tiles between the TP_SIZE devices of the line over the
1D fabric.  ``Transport`` and ``transport_mesh_program`` build the per-device program (one transport core per fabric
link, a line multicast by hop count); ``Line`` / ``LineSemaphores`` hold the per-mesh line and its global semaphores;
the phase kernels are ``kernels/transport.cpp``, ``kernels/transport2.cpp`` and ``kernels/transport_phase.h``.
row_gather, row_gather2, row_reduce, gr_recip_last, qsa_trim and moe_experts build their programs on it.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

import ttnn

from ...contracts import TP_SIZE
from .. import gr_read
from .. import program as fp

NAME = "gr_fold"
TP_AXIS = 1
TRANSPORT_KERNEL = fp.kernel_source(NAME, "transport.cpp")
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
EXTRA_PHASES = ("rowgather",)  # the semaphore pairs of other programs on the line transport (row_gather)
STATS_SCRATCH_CB = 3  # beside the stats program's CBs 0, 1, 2, 16
PARTIALS_SCRATCH_CB = 11  # beside normalize_down's CBs 0-10, 16, 17
# the transport cores per phase, one per link: the stats phase next to the four stats cores, the partials phase
# clear of normalize_down's 6x2 worker rectangle and its producers on row 2
TRANSPORT = {
    "stats": (ttnn.CoreCoord(gr_read.BRANCHES, 0), ttnn.CoreCoord(gr_read.BRANCHES + 1, 0)),
    "partials": (ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 0)),
}
PARTIALS_SEMAPHORES = (1, 2, 3)  # normalize_down's in0 multicast owns program semaphore 0
# the stage-(d) program (stats + normalize_down as one program): the partials transport cores run both phases on one
# connection per link (kernels/transport2.cpp; a link's worker sender channel is one per direction), the stats phase
# with its own scratch semaphore; the gate is the norm cores' semaphore their reader waits on before streaming the
# gathered stats
# row-trimmed transport (transport_phase.h): 0 whole tiles, 1 one packet spanning the valid rows, 2 one packet per chunk, 3 packed (all tiles in one packet into the peer L1, unpacked there)
ROW_MODE = int(os.environ.get("QWEN38_GR_ROW_MODE", "3"))
# QWEN38_GR_NOBARRIER=1 (default 0): the two-phase packed transport (row mode 4) skips the barrier round trip; it lands
# the packets in double-buffered slot sets (call parity) of a persistent L1 buffer on the transport cores, one data
# semaphore per parity, a persistent per-phase call counter in that buffer (transport_phase.h: NOBARRIER)
NOBARRIER = os.environ.get("QWEN38_GR_NOBARRIER", "0") == "1"
LAND_HEADER_BYTES = 16  # the call counter (one word) ahead of the two slot sets, keeps the slots 16 B aligned
TRANSPORT2_KERNEL = fp.kernel_source(NAME, "transport2.cpp")
FRONT_STATS_SCRATCH_SEM = 4  # on the transport cores, beside the partials' (go, scratch, done) = 1, 2, 3
FRONT_STATS_READY = 5  # on the norm cores, beside the in0 multicast's 0
FRONT_STATS_CBS = (
    20,
    21,
    22,
    23,
)  # the stats phase's scaler (fp32), x^2 (fp32), stats tile (bf16), transport scratch (bf16)
# The caches below are keyed by id(mesh): a strong reference to every mesh they have seen keeps a closed mesh's id from being reused by
# a new mesh (a stale entry would hand out freed L1 addresses).  One mesh per process in the model; many in the tests.
_MESH_KEEP: list = []
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
            for phase in PHASES + EXTRA_PHASES
        }
        self.addresses = {
            phase: (int(ttnn.get_global_semaphore_address(barrier)), int(ttnn.get_global_semaphore_address(data)))
            for phase, (barrier, data) in self.handles.items()
        }
        # NOBARRIER: the odd-parity data semaphore of each two-phase transport phase (created only with the flag, so the
        # default allocates exactly what it did before)
        self.data1_handles = (
            {phase: ttnn.create_global_semaphore(mesh, cores, 0) for phase in PHASES} if NOBARRIER else {}
        )
        self.data1_addresses = {
            phase: int(ttnn.get_global_semaphore_address(sem)) for phase, sem in self.data1_handles.items()
        }
        self.lands: dict = {}  # (phase, slot bytes, links) -> (persistent L1 landing tensor, its address)

    def pair(self, phase: str) -> tuple[int, int]:
        return self.addresses[phase]

    def land(self, mesh, phase: str, slot: int, cores) -> tuple[int, int]:
        """(odd data semaphore address, landing buffer address) of a NOBARRIER phase: a persistent L1 buffer, zeroed,
        on every transport core of the links, laid out [call counter, 16 B][slot set 0: TP_SIZE slots][slot set 1]; it
        outlives every program (a peer's packet may land while this device runs another program, so the landing
        slots cannot be program CB memory) and its counter is never reset (the call parity survives trace replays).
        One buffer per (phase, slot bytes): a program shape owns its counter."""

        key = (phase, slot, len(cores))
        if key not in self.lands:
            words = (LAND_HEADER_BYTES + 2 * TP_SIZE * slot) // 4
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
            )
            tensor = ttnn.zeros(
                [len(cores), words], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=memory
            )
            self.lands[key] = (tensor, int(tensor.buffer_address()))
        return self.data1_addresses[phase], self.lands[key][1]


def line_semaphores(mesh) -> LineSemaphores:
    key = id(mesh)
    if key not in _SEMAPHORES:
        _MESH_KEEP.append(mesh)
        _SEMAPHORES[key] = LineSemaphores(mesh)
    return _SEMAPHORES[key]


class Line:
    """The TP line as the fabric sees it: rank r -> (mesh coordinate, fabric node id), checked once per mesh: the
    mesh is the model's ``(1, TP_SIZE)`` with the line along axis 1, every neighbour pair is one hop in a direction
    consistent along the line (forward one way, backward the other), the degree histogram is {1: 2, 2: TP_SIZE - 2},
    and the link indices usable towards each neighbour are recorded (``links`` = the count every pair offers)."""

    def __init__(self, mesh) -> None:
        shape = tuple(int(v) for v in mesh.shape)
        if shape != (1, TP_SIZE):
            raise ValueError(f"the line transport needs the (1, {TP_SIZE}) mesh, got {shape}")
        self.coordinates = [ttnn.MeshCoordinate(0, rank) for rank in range(TP_SIZE)]
        self.nodes = [mesh.get_fabric_node_id(coordinate) for coordinate in self.coordinates]
        forward = [
            ttnn.get_eth_forwarding_direction(mesh, self.nodes[r], self.nodes[r + 1]) for r in range(TP_SIZE - 1)
        ]
        backward = [
            ttnn.get_eth_forwarding_direction(mesh, self.nodes[r + 1], self.nodes[r]) for r in range(TP_SIZE - 1)
        ]
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
                if (
                    other != rank
                    and ttnn.get_eth_forwarding_direction(mesh, self.nodes[rank], self.nodes[other]) is not None
                ):
                    degree[rank] += 1
        # every rank can forward to every other on a line; the immediate-neighbour count is what the transport uses
        neighbours = {rank: (rank > 0) + (rank + 1 < TP_SIZE) for rank in range(TP_SIZE)}
        if Counter(neighbours.values()) != Counter({1: 2, 2: TP_SIZE - 2}) or any(
            degree[r] != TP_SIZE - 1 for r in range(TP_SIZE)
        ):
            raise RuntimeError(
                f"line degree histogram is not {{1: 2, 2: {TP_SIZE - 2}}}: "
                f"neighbours {neighbours} reachable {dict(degree)}"
            )
        link_counts = {
            len(ttnn.get_forwarding_link_indices(mesh, self.nodes[a], self.nodes[b]))
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
        _MESH_KEEP.append(mesh)
        _LINES[key] = Line(mesh)
    return _LINES[key]


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
    kernel: Any = None  # single-phase programs: a kernel source other than transport.cpp (same runtime args)
    extra_args: tuple = ()  # with ``kernel``: compile args after the page strides, before the accessors
    pack_cb: int = 0  # mode 3: CB id of the raw L1 buffer (landing slots + staging) on the transport cores
    rows: int = 0  # valid rows per tile (two-phase transport only): 0 or >= TILE sends whole tiles


def _row_mode(t, links, mode=None):
    """(mode, slot bytes) of one phase of a two-phase program: 0 = whole tiles (rows unknown or a full tile),
    ROW_MODE otherwise; mode 3 falls back to mode 1 (one span packet per tile) when the core's packed tiles exceed
    one fabric packet payload (ttnn.get_tt_fabric_max_payload_size_bytes)."""

    want = ROW_MODE if mode is None else mode
    if not 0 < t.rows < fp.TILE or want == 0:
        return 0, 0
    if want != 3:
        return want, 0
    tile_bytes = fp.TILE_BYTES[t.out.dtype]
    face, row = tile_bytes // 4, tile_bytes // 64
    packed = 2 * t.rows * row if t.rows <= 16 else 2 * face + 2 * (t.rows - 16) * row
    my_tiles = -(-t.tiles // links)
    if my_tiles * packed > int(ttnn.get_tt_fabric_max_payload_size_bytes()):
        return 1, 0
    return 3, -(-(my_tiles * packed) // 16) * 16


def ring_route() -> bool:
    """Every custom fabric exchange routes over the 1x8 ring (wrap connections rank 0 <-> rank 7, ring ranges in
    ``LineSender::open``) when ``contracts.ring_enabled()`` (``QWEN38_RING``)."""
    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import ring_enabled

    return ring_enabled()


def transport_mesh_program(
    mesh, transports, *, semaphores, cbs, kernels=lambda rank: [], links: int = 1, ring: bool | None = None
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
    modes = [_row_mode(t, links) for t in transports] if len(transports) == 2 else []
    nobarrier = NOBARRIER and bool(modes) and all(m == 3 for m, _ in modes)
    tail = []  # NOBARRIER: compile args after the accessors: both phases' odd data semaphore, both landing buffers
    if nobarrier:
        modes = [(4, slot) for _, slot in modes]
        lands = [line_semaphores(mesh).land(mesh, t.phase, slot, cores) for t, (_, slot) in zip(transports, modes)]
        tail = [d for d, _ in lands] + [a for _, a in lands]
    cbs = list(cbs)
    for t, (mode, slot) in zip(transports, modes):
        if mode in (3, 4):  # mode 3: RING landing slots + one staging slot per RISC, one raw L1 buffer on the transport
            # cores; mode 4: only the two staging slots (the landing slots are the persistent buffer's)
            cbs.append(
                fp.cb_descriptor(
                    t.pack_cb,
                    gr_read.BF16,
                    (2 if mode == 4 else TP_SIZE + 2) * slot,
                    1,
                    ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]),
                )
            )
    ring = ring_route() if ring is None else ring
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
                    *first.extra_args,
                ]
            else:
                args = [go, TP_SIZE, done, i, links]
                for t in transports:
                    args += [t.scratch_cb, t.tiles, t.source, t.semaphore_ids[1], *t.page_strides]
                args += [t.rows if 0 < t.rows < fp.TILE else fp.TILE for t in transports]
                args += [m for m, _ in modes]
                args += [t.pack_cb for t in transports]
                args += [slot for _, slot in modes]
            for t in transports:
                args += fp.accessor_args(t.out) + fp.accessor_args(t.local)
            args += tail
            compile_args.append(args)
        kernel = (first.kernel or TRANSPORT_KERNEL) if len(transports) == 1 else TRANSPORT2_KERNEL

        def transport(build, per_link_args):
            return [
                build(
                    kernel,
                    ttnn.CoreRangeSet([ttnn.CoreRange(cores[i], cores[i])]),
                    compile_args[i],
                    [(cores[i], base + list(per_link_args[i]))],
                    defines=[("QWEN38_RING_ROUTE", "1")] if ring else (),
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
            ttnn.setup_fabric_connection(mesh, node, geometry.nodes[(rank - 1) % TP_SIZE], i, seed, cores[i])
            if (rank > 0 or ring)
            else []
            for i in range(links)
        ]
        forward = [
            (
                ttnn.setup_fabric_connection(mesh, node, geometry.nodes[(rank + 1) % TP_SIZE], i, seed, cores[i])
                if (rank + 1 < TP_SIZE or ring)
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
