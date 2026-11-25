# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A bare all-to-all over the 2D fabric (kernels/a2a_*.cpp): every chip sends ROWS_PER_DST rows of PB bytes to every
other chip as fabric unicasts, nothing else - the traffic floor of an expert combine.

A chip's rows to a destination leave by their first hop (the fabric routes Y first, then X): one sender core a
(direction, link), each with its own fabric connection, the rows to its destinations interleaved; a destination's
rows land round-robin on NLAND cores (overwriting one slot: traffic only); each sender's last row to a destination
bumps that chip's counter, and one receiver core a chip waits for all of them.
"""

from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.combine_lab.bench_util import crs, kpath

FAB_DEFS = [("FABRIC_2D", "1")]


def router_cfg(payload):
    c = ttnn._ttnn.fabric.FabricRouterConfig()
    c.max_packet_payload_size_bytes = payload
    return c


def node(mesh, rc):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*rc))


def first_hop(src, dst):
    """The neighbour a row from src to dst leaves through (dimension order: Y (mesh rows) first, then X)."""
    (r, c), (dr, dc) = src, dst
    if dr != r:
        return (r + (1 if dr > r else -1), c)
    return (r, c + (1 if dc > c else -1))


# This LoudBox (bh-lb-09, P150s): the fabric's ethernet channel ids count the unharvested channels; channels 5 / 8 are
# fused (cluster descriptor eth_harvesting_mask 288) - channel -> physical NoC x of its ethernet core (row y 1)
BH_ETH_X = (1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10)
ETH_X = [x for i, x in enumerate(BH_ETH_X) if i not in (5, 8)]
# the 12 x 10 grid's logical columns -> physical NoC x (Tensix columns 1-7, 10-16; x 10 harvested, the last enabled
# column the dispatch's) - logical row r = NoC y r + 2
PHYS_X = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]
LOGICAL_COL = {x: c for c, x in enumerate(PHYS_X)}


def eth_channel(mesh, src, dst, link):
    a = list(
        ttnn.setup_fabric_connection(
            mesh_device=mesh,
            src_fabric_node_id=node(mesh, src),
            dst_fabric_node_id=node(mesh, dst),
            link_idx=link,
            program_descriptor=ttnn.ProgramDescriptor(),
            worker_core=ttnn.CoreCoord(0, 0),
        )
    )
    return a[0]


def ws_sender(ch):
    """The sender core for a connection on ethernet channel ch: directly under its ethernet core (logical row 0) on
    NOC1; an ethernet core over a column without a Tensix core: a NOC0 sender from the bottom row, west of it."""
    x = ETH_X[ch]
    if x in LOGICAL_COL:
        return (LOGICAL_COL[x], 0), 1
    west = max(c for c in PHYS_X if c < x)
    return (LOGICAL_COL[west], 9), 0


def ws_landing(ch):
    """A flow's landing core: the bottom row of the column of the router it arrives through (the router's NOC1 landing
    write: y 1 -> 0 -> 11, no row travel); a column without a Tensix core: the bottom row west of it."""
    x = ETH_X[ch]
    if x in LOGICAL_COL:
        return (LOGICAL_COL[x], 9)
    return (LOGICAL_COL[max(c for c in PHYS_X if c < x)], 9)


@dataclass
class A2ACfg:
    PB: int = 12320  # bytes a row (GLM-5.3: 6144 bf16 + the combine's 32 B trailer)
    rows_per_dst: int = 768  # rows a chip sends each other chip (GLM-5.3 16 experts M 384: 6144 rows over 8 chips)
    links: int = 2
    nland: int = 20  # landing cores a chip
    land: list = field(default_factory=lambda: [(x, y) for y in range(5) for x in range(2, 6)])
    senders: list = field(default_factory=lambda: [(0, y) for y in range(6)])  # (one a (direction, link))
    counter: tuple = (6, 9)
    placement: str = "simple"  # "simple" (the cores above, NCRISC NOC0) / "ws" (ws_expert FABRIC_FINDINGS placement)


def build(mesh, cfg: A2ACfg):
    R, C = tuple(mesh.shape)
    chips = [(r, c) for r in range(R) for c in range(C)]
    vxy = lambda x, y: mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    pk = lambda c: vxy(*c).x | vxy(*c).y << 8
    # flows: (chip, neighbour, link) -> destinations
    flows = {}
    for s in chips:
        for d in chips:
            if d != s:
                for l in range(cfg.links):
                    flows.setdefault((s, first_hop(s, d), l), []).append(d)
    want = {c: sum(1 for (s, nb, l), ds in flows.items() for d in ds if d == c) for c in chips}
    rows = cfg.rows_per_dst // cfg.links

    def last_hop_from(s, d):  # the neighbour of d a row from s arrives from (Y first, then X)
        if s[1] != d[1]:
            return (d[0], d[1] + (-1 if s[1] < d[1] else 1))
        return (d[0] + (-1 if s[0] < d[0] else 1), d[1])

    sender, landing = {}, {}  # (chip, nb, l) -> (core, noc); (src flow, dst) -> landing core
    counter = cfg.counter
    for (s, nb, l), ds in flows.items():
        i = [k for k in flows if k[0] == s].index((s, nb, l))
        if cfg.placement == "ws":
            sender[(s, nb, l)] = ws_sender(eth_channel(mesh, s, nb, l))
            for d in ds:
                landing[((s, nb, l), d)] = ws_landing(eth_channel(mesh, d, last_hop_from(s, d), l))
            counter = (10, 5)
        else:
            sender[(s, nb, l)] = (cfg.senders[i], 0)
            for j, d in enumerate(ds):
                landing[((s, nb, l), d)] = cfg.land[(i * 7 + j) % cfg.nland]
    cores = sorted({c for c, _ in sender.values()} | set(landing.values()) | {counter})
    mem = lambda n: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(cores), [n, 16], ttnn.ShardOrientation.ROW_MAJOR),
    )
    rep = ttnn.ReplicateTensorToMesh(mesh)
    nrow = -(-cfg.PB // 64)
    buf = ttnn.from_torch(
        torch.zeros(len(cores) * nrow, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh, memory_config=mem(nrow), mesh_mapper=rep,
    )  # fmt: skip
    cnt = ttnn.from_torch(
        torch.zeros(len(cores), 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh, memory_config=mem(1), mesh_mapper=rep,
    )  # fmt: skip
    mpd = ttnn.MeshProgramDescriptor()
    n_senders = {}
    for ch in chips:
        mine = [(k, ds) for k, ds in flows.items() if k[0] == ch]
        n_senders[ch] = len(mine)
        groups = {}  # (ndst, noc) -> [(core, nb, l, ds)]
        for k, ds in mine:
            core, nc = sender[k]
            groups.setdefault((len(ds), nc), []).append((core, k, ds))
        kernels = []
        for (ndst, nc), group in groups.items():
            rt = ttnn.RuntimeArgs()
            for core, k, ds in group:
                args = [int(node(mesh, ch).mesh_id), pk(counter)]
                for d in ds:
                    args += [int(node(mesh, d).chip_id), pk(landing[(k, d)])]
                rt[core[0]][core[1]] = args
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("a2a_send.cpp"),
                    core_ranges=crs([g[0] for g in group]),
                    compile_time_args=[cfg.PB, ndst, rows, 0, buf.buffer_address(), cnt.buffer_address()],
                    runtime_args=rt,
                    defines=FAB_DEFS,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0 if nc == 1 else ttnn.DataMovementProcessor.RISCV_1,
                        noc=ttnn.NOC.NOC_1 if nc == 1 else ttnn.NOC.NOC_0,
                    ),
                )
            )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("a2a_recv.cpp"),
                core_ranges=crs([counter]),
                compile_time_args=[want[ch], cnt.buffer_address()],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            )
        )
        p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
        for ki, group in enumerate(groups.values()):
            for core, k, ds in group:
                p.kernels[ki].runtime_args[core[0]][core[1]].extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, ch),
                        dst_fabric_node_id=node(mesh, k[1]),
                        link_idx=k[2],
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(*core),
                    )
                )
        mc = ttnn.MeshCoordinate(*ch)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    bytes_out = {ch: (len(chips) - 1) * cfg.rows_per_dst * cfg.PB for ch in chips}
    return mpd, [buf, cnt], bytes_out, n_senders
