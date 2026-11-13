# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert groups with the weights landing over the fabric (LoudBox 2 x 4, FABRIC_2D).

Step 1 (this file): one receiving chip R runs the 80-core group program (ws_group, no injectors); its neighbours
only send weights. Each sender chip has one sender core per link toward R (2 links a neighbour): the BRISC reads the
target cores' slices from the chip's DRAM, the NCRISC sends them as CHT-tile packets straight into R's 2-slot rings,
the last packet of a slice fused with an atomic inc of the core's warr word. R's cores report freed slots to a relay
core on R, which forwards each count over the fabric to the sender core's credit word (same tensor address).
Target cores are dealt round-robin over the connections (core c = g * GC + j -> connection c % nconn).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_group import GCfg, build

FAB_DEFS = [("FABRIC_2D", "1")]  # (ttnn ops get it from the op infrastructure; generic_op kernels set it)


@dataclass
class FCfg:
    recv: tuple = (0, 1)  # mesh coordinate of the receiving chip
    links: int = 2  # sender connections a neighbour
    CHT: int = 24  # tiles a packet (24 x 576 B bfp4 = 13.8 KB)
    SCB: int = 4  # sender staging (packets)
    payload: int = 15232  # fabric router max payload
    relay: tuple = (10, 9)  # logical core of the credit relay on R
    send_cores: tuple = ((9, 9), (10, 8))  # logical cores of a sender chip's connections (link 0, 1)


def neighbours(mesh_shape, rc):
    r, c = rc
    out = []
    for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < mesh_shape[0] and 0 <= cc < mesh_shape[1]:
            out.append((rr, cc))
    return out


def node(mesh, rc):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*rc))


def build_fab(mesh, cfg: GCfg, fc: FCfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t):
    NC = cfg.G * cfg.GC
    nbrs = neighbours(tuple(mesh.shape), fc.recv)
    conns = [(nb, l) for nb in nbrs for l in range(fc.links)]  # (sender chip, link)
    targets = {k: [c for c in range(NC) if c % len(conns) == k] for k in range(len(conns))}
    # credit words: on the relay (R) and on every sender core (same address); first, at the top of L1
    cred_cores = [fc.relay] + list(fc.send_cores[: fc.links])
    icred = sharded_l1(mesh, cred_cores, [NC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    from models.demos.ws_expert.ws_group import group_cores

    gcores = [c for g in range(cfg.G) for c in group_cores(cfg, g)]  # index c = g * GC + j
    assert fc.relay not in gcores and not set(fc.send_cores) & set(gcores)
    rrt = ttnn.RuntimeArgs()
    rrt[fc.relay[0]][fc.relay[1]] = [len(nbrs)]
    relay_kd = ttnn.KernelDescriptor(
        kernel_source=kpath("wf_relay.cpp"),
        core_ranges=crs([fc.relay]),
        compile_time_args=[NC, cfg.GC, cfg.G, cfg.nexp, icred.buffer_address()],
        runtime_args=rrt,
        defines=FAB_DEFS,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    rprog, io, st, ex = build(
        mesh, cfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t, fab=dict(relay=fc.relay, icred=icred, kernels=[relay_kd])
    )
    # relay RT: nconn (one connection a sender chip, link 0), per connection: mesh, chip, fabric args; then the map
    rx, ry = fc.relay
    rk = len(rprog.kernels) - 1
    rt = rprog.kernels[rk].runtime_args[rx][ry]
    for nb in nbrs:
        d = node(mesh, nb)
        rt.extend([int(d.mesh_id), int(d.chip_id)])
        rt.extend(
            ttnn.setup_fabric_connection(
                mesh_device=mesh,
                src_fabric_node_id=node(mesh, fc.recv),
                dst_fabric_node_id=d,
                link_idx=0,
                program_descriptor=rprog,
                worker_core=ttnn.CoreCoord(rx, ry),
            )
        )
    conn_of = {c: k for k, cs in targets.items() for c in cs}
    send_noc = [noc(mesh, *sc) for sc in fc.send_cores[: fc.links]]
    for c in range(NC):
        k = conn_of[c]
        nb, l = conns[k]
        sx, sy = send_noc[l]
        rt.extend([nbrs.index(nb) | sx << 8 | sy << 16])

    # sender chips
    KT, KTD, DC = cfg.KT, cfg.PAIRS, cfg.DCOLS
    sprogs = {}
    for nb in nbrs:
        scores = list(fc.send_cores[: fc.links])
        rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for l, (x, y) in enumerate(scores):
            k = conns.index((nb, l))
            tl = [c | noc(mesh, *gcores[c])[0] << 8 | noc(mesh, *gcores[c])[1] << 16 for c in targets[k]]
            rd_rt[x][y] = [wgu_t.buffer_address(), wd_t.buffer_address(), len(tl)] + tl
            d = node(mesh, fc.recv)
            sd_rt[x][y] = [int(d.mesh_id), int(d.chip_id), len(tl)] + tl
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=kpath("wf_sreader.cpp"),
                core_ranges=crs(scores),
                compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.PAIRS, cfg.HT, fc.CHT, cfg.G, cfg.GC]
                + list(ttnn.TensorAccessorArgs(wgu_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args()),
                runtime_args=rd_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("wf_sender.cpp"),
                core_ranges=crs(scores),
                compile_time_args=[cfg.nexp, KT, KTD, DC, fc.CHT, cfg.G, cfg.GC]
                + [ex["wgu_ring"].buffer_address(), ex["wd_ring"].buffer_address(), ex["warr"].buffer_address()]
                + [icred.buffer_address()],
                runtime_args=sd_rt,
                defines=FAB_DEFS,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            ),
        ]
        sp = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(0, cfg.w_dt, fc.SCB * fc.CHT, crs(scores))])
        for l, (x, y) in enumerate(scores):
            sp.kernels[1].runtime_args[x][y].extend(
                ttnn.setup_fabric_connection(
                    mesh_device=mesh,
                    src_fabric_node_id=node(mesh, nb),
                    dst_fabric_node_id=node(mesh, fc.recv),
                    link_idx=l,
                    program_descriptor=sp,
                    worker_core=ttnn.CoreCoord(x, y),
                )
            )
        sprogs[nb] = sp
    mpd = ttnn.MeshProgramDescriptor()
    rc = ttnn.MeshCoordinate(*fc.recv)
    mpd[ttnn.MeshCoordinateRange(rc, rc)] = rprog
    for nb, sp in sprogs.items():
        c = ttnn.MeshCoordinate(*nb)
        mpd[ttnn.MeshCoordinateRange(c, c)] = sp
    io = io + [icred]
    return mpd, io, st
