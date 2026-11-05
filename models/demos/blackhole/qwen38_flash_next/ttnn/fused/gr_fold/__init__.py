# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gr_fold``: the line transports the fused GR read's front (``gr_recip_last``) runs its two gathers on.

One transport core per fabric link sends this device's tiles to the other devices of the TP4 line over the 1D fabric,
into the DRAM pages the chain's all-gather writes (page ``4b + d`` for the stats, ``12d + t`` for the partials), so a
gather phase is bitwise by construction. ``kernels/transport_phase.h`` states the semaphore protocol; ``line`` and
``line_semaphores`` are per mesh and must be created before any trace capture.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import ttnn

from .. import program as fp

NAME = "gr_fold"
TP_SIZE = 4
TP_AXIS = 1
SEM_GO, SEM_SCRATCH, SEM_DONE = (
    0,
    1,
    2,
)  # program-local, allocated first: barrier passed (BRISC -> NCRISC), producers' tiles landed, NCRISC done
SOURCE_TENSOR, SOURCE_PRODUCERS = 0, 1
PHASES = (
    "stats",
    "partials",
)  # one (barrier, data) global semaphore pair per gather phase, never cycled (kernels/transport_phase.h)
PARTIALS_SCRATCH_CB = 11  # beside normalize_down's CBs 0-10, 16, 17
# the transport cores, one per link: clear of normalize_down's 6x2 worker rectangle and its producers on row 2
TRANSPORT = {"partials": (ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 0))}
PARTIALS_SEMAPHORES = (1, 2, 3)  # normalize_down's in0 multicast owns program semaphore 0
# the front's one program: the transport cores run both phases on one connection per link (kernels/transport2.cpp),
# the stats phase with its own scratch semaphore; the norm cores' reader waits on FRONT_STATS_READY
TRANSPORT2_KERNEL = fp.kernel_source(NAME, "transport2.cpp")
FRONT_STATS_SCRATCH_SEM = 4  # on the transport cores, beside the partials' (go, scratch, done) = 1, 2, 3
FRONT_STATS_READY = 5  # on the norm cores, beside the in0 multicast's 0
FRONT_STATS_CBS = (
    20,
    21,
    22,
    23,
)  # the stats phase's scaler (fp32), x^2 (fp32), stats tile (bf16), transport scratch (bf16)
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
    """The TP line as the fabric sees it: rank r -> (mesh coordinate, fabric node id), checked once per mesh.

    The mesh must be the model's ``(1, 4)`` line along axis 1 with one hop per neighbour pair in a consistent
    direction; ``links`` is the link count every neighbour pair offers."""

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


@dataclass(frozen=True)
class Transport:
    """One gather phase inside a program: tiles of ``local`` (SOURCE_TENSOR) or of producer cores (SOURCE_PRODUCERS)
    land in ``out`` at page ``t * page_strides[0] + rank * page_strides[1]``, moved by ``cores[:links]`` (one per
    link, tiles dealt round robin) with the program-local semaphore ids (go, scratch, done); the transport cores
    raise ``consumer_sem`` on each of ``consumers`` (NoC x, y) once every peer's tile has landed; ``delays`` = {rank:
    (before_arrive, after_reset)} spin cycles for skew testing (the first transport's, in a two-phase program)."""

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
    """One ProgramDescriptor per mesh coordinate: ``kernels(rank)`` plus each link's transport reader (NCRISC,
    backward) and writer (BRISC, forward).

    Both ``Transport`` phases run ``kernels/transport2.cpp`` on the same cores and connection (a link has one worker
    sender channel per direction). The program's own ``semaphores`` (fixed ids) precede the connection semaphores
    ``setup_fabric_connection`` adds (backward, then forward), so ids are stable across builds; that call mutates the
    descriptor and returns the runtime args, so the transport kernels are built twice."""

    geometry = line(mesh)
    if len(transports) != 2:
        raise ValueError(f"a program carries two transports (one per phase), got {len(transports)}")
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
            args = [go, TP_SIZE, done, i, links]
            for t in transports:
                args += [t.scratch_cb, t.tiles, t.source, t.semaphore_ids[1], *t.page_strides]
            for t in transports:
                args += fp.accessor_args(t.out) + fp.accessor_args(t.local)
            compile_args.append(args)

        def transport(build, per_link_args):
            return [
                build(
                    TRANSPORT2_KERNEL,
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
