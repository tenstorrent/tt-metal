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
    SCB: int = 8  # sender staging (packets)
    RD: int = 4  # sender DRAM reads in flight (chunks)
    payload: int = 15232  # fabric router max payload
    relay: tuple = (10, 9)  # logical core of the credit relay on R
    send_cores: tuple = ((9, 9), (10, 8))  # logical cores of a sender chip's connections (link 0, 1)


def slice_layout(cfg: GCfg, CHT, wg, wu, wd):
    """The senders' DRAM copy of the weights: per expert e and group core j, the core's slice (gate/up pair j
    [KT][2], then down columns [DCOLS j, DCOLS (j + 1)): [KTD][DCOLS]) in CHT-tile chunks q = (e GC + j) NCH + u, chunk q
    stored as pages (q // 8) CHT 8 + q % 8 + 8 k (k < CHT): CHT consecutive tiles of one DRAM bank (one read a chunk).
    Returns a [pages * 32, 32] torch tensor (tile layout rows)."""
    E, KT, KTD, DC, P, HT = wg.shape[0], cfg.KT, cfg.PAIRS, cfg.DCOLS, cfg.PAIRS, cfg.HT
    from models.demos.ws_expert.ws_expert import interleave_gu

    tiles = []
    for e in range(E):
        gu = interleave_gu(wg[e], wu[e]).reshape(KT, 32, 2 * P, 32).permute(0, 2, 1, 3)  # [KT, 2P, 32, 32]
        dn = wd[e].reshape(KTD, 32, HT, 32).permute(0, 2, 1, 3)  # [KTD, HT, 32, 32]
        for j in range(cfg.GC):
            tiles.append(gu[:, 2 * j : 2 * j + 2].reshape(-1, 32, 32))
            tiles.append(dn[:, DC * j : DC * (j + 1)].reshape(-1, 32, 32))
    t = torch.cat(tiles, 0)  # slice-major: [E GC WTOT, 32, 32]
    nq = t.shape[0] // CHT
    nq_pad = -(-nq // 8) * 8
    t = torch.cat([t, torch.zeros((nq_pad - nq) * CHT, 32, 32)], 0).reshape(nq_pad, CHT, 32, 32)
    q = torch.arange(nq_pad)
    pages = ((q // 8) * CHT * 8 + q % 8).unsqueeze(1) + 8 * torch.arange(CHT).unsqueeze(0)  # [nq, CHT]
    out = torch.empty(nq_pad * CHT, 32, 32)
    out[pages.flatten()] = t.reshape(-1, 32, 32)
    return out.reshape(-1, 32)


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


def build_fab(mesh, cfg: GCfg, fc: FCfg, x_t, idx_t, counts_t, ws_t, out_t):
    """ws_t: the senders' weight slices (slice_layout), replicated."""
    wgu_t = wd_t = ws_t
    NC = cfg.G * cfg.GC
    nbrs = neighbours(tuple(mesh.shape), fc.recv)
    conns = [(nb, l) for nb in nbrs for l in range(fc.links)]  # (sender chip, link)
    targets = {k: [c for c in range(NC) if c % len(conns) == k] for k in range(len(conns))}
    # credit words: on the relay (R) and on every sender core (same address); first, at the top of L1
    cred_cores = [fc.relay] + list(fc.send_cores[: fc.links])
    icred = sharded_l1(mesh, cred_cores, [NC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    sst = sharded_l1(mesh, cred_cores, [1, 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)  # sender timers
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
            rd_rt[x][y] = [ws_t.buffer_address(), 0, len(tl)] + tl
            d = node(mesh, fc.recv)
            sd_rt[x][y] = [int(d.mesh_id), int(d.chip_id), len(tl)] + tl
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=kpath("wf_sreader.cpp"),
                core_ranges=crs(scores),
                compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.PAIRS, cfg.HT, fc.CHT, cfg.G, cfg.GC]
                + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                runtime_args=rd_rt,
                defines=[
                    ("RD_DEPTH", str(fc.RD)),
                    ("SCB_CHUNKS", str(fc.SCB)),
                    ("SST_ADDR", str(sst.buffer_address())),
                ],
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
                defines=FAB_DEFS + [("SST_ADDR", str(sst.buffer_address()))],
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
    io = io + [icred, sst]
    return mpd, io, st, sst


# ---------------------------------------------------------------------------------------------------------------
# Step 2: every chip runs the groups on its own tokens; expert e is owned by chip e % NCHIP, which lands its slices on
# every chip: one core a fabric branch (E-only, W-only, N/S with E/W teeth: 2D multicast) plus a local lander (NoC).
# Each chip's relay forwards freed slots (link 1 of each neighbour) to the owner-side senders that cover the chip.
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ACfg:
    CHT: int = 24
    SCB: int = 8
    RD: int = 4
    relay: tuple = (10, 9)
    local: tuple = (9, 9)  # local lander
    branch: tuple = ((10, 8), (9, 8), (10, 7))  # E, W, N/S-with-teeth sender cores


def branches(mesh_shape, rc):
    """[(name, first-hop neighbour, (e, w, n, s), covered chips)] of chip rc's multicast branches (2-row mesh)."""
    R, Cn = mesh_shape
    r, c = rc
    assert R == 2
    out = []
    if c < Cn - 1:
        out.append(("E", (r, c + 1), (Cn - 1 - c, 0, 0, 0), [(r, cc) for cc in range(c + 1, Cn)]))
    if c > 0:
        out.append(("W", (r, c - 1), (0, c, 0, 0), [(r, cc) for cc in range(c)]))
    r2 = 1 - r
    ns = (Cn - 1 - c, c, 1, 0) if r2 < r else (Cn - 1 - c, c, 0, 1)  # N toward row 0, S toward row + 1
    out.append(("NS", (r2, c), ns, [(r2, cc) for cc in range(Cn)]))
    return out


def build_all(mesh, cfg: GCfg, ac: ACfg, x_t, idx_t, counts_t, ws_t, out_t):
    R, Cn = tuple(mesh.shape)
    NCHIP = R * Cn
    NC = cfg.G * cfg.GC
    chip = lambda rc: rc[0] * Cn + rc[1]
    bcore = dict(zip(("E", "W", "NS"), ac.branch))
    send_cores = [ac.local] + list(ac.branch)
    icred = sharded_l1(mesh, [ac.relay] + send_cores, [NCHIP * NC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    sst = sharded_l1(mesh, send_cores, [1, 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    from models.demos.ws_expert.ws_group import group_cores

    gcores = [c for g in range(cfg.G) for c in group_cores(cfg, g)]
    assert not set([ac.relay] + send_cores) & set(gcores)
    gprog, io, st, ex = build(mesh, cfg, x_t, idx_t, counts_t, ws_t, ws_t, out_t, fab=dict(relay=ac.relay, icred=icred))
    tl = [c | noc(mesh, *gcores[c])[0] << 8 | noc(mesh, *gcores[c])[1] << 16 for c in range(NC)]
    KT, KTD, DC = cfg.KT, cfg.PAIRS, cfg.DCOLS
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            rc, me = (r, c), chip((r, c))
            brs = branches((R, Cn), rc)
            nbrs = neighbours((R, Cn), rc)
            # relay
            rrt = ttnn.RuntimeArgs()
            rrt[ac.relay[0]][ac.relay[1]] = [me, len(nbrs)]
            relay = ttnn.KernelDescriptor(
                kernel_source=kpath("wf_relay_all.cpp"),
                core_ranges=crs([ac.relay]),
                compile_time_args=[NC, cfg.GC, cfg.G, cfg.nexp, icred.buffer_address(), NCHIP, mesh_id],
                runtime_args=rrt,
                defines=FAB_DEFS,
                config=dm(0),
            )
            # owner-side readers (all sender cores) + local lander + branch senders
            rd_rt, ln_rt, br_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for x, y in [ac.local] + [bcore[b[0]] for b in brs]:
                rd_rt[x][y] = [ws_t.buffer_address(), me, NC] + tl
            ln_rt[ac.local[0]][ac.local[1]] = [mesh_id, me, 0, 0, 0, 0, 1 << me, NC] + tl
            for name, nb, (e, w, n, s), cov in brs:
                x, y = bcore[name]
                br_rt[x][y] = [mesh_id, me, e, w, n, s, sum(1 << chip(k) for k in cov), NC] + tl
            scores = [ac.local] + [bcore[b[0]] for b in brs]
            sdefs = [("SST_ADDR", str(sst.buffer_address()))]
            ct_send = [cfg.nexp, KT, KTD, DC, ac.CHT, cfg.G, cfg.GC]
            ct_send += [ex["wgu_ring"].buffer_address(), ex["wd_ring"].buffer_address(), ex["warr"].buffer_address()]
            ct_send += [icred.buffer_address(), NC, NCHIP, me]
            kernels = list(gprog.kernels) + [
                relay,
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_sreader.cpp"),
                    core_ranges=crs(scores),
                    compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.PAIRS, cfg.HT, ac.CHT, cfg.G, cfg.GC]
                    + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                    runtime_args=rd_rt,
                    defines=[("RD_DEPTH", str(ac.RD)), ("SCB_CHUNKS", str(ac.SCB)), ("NOWN", str(NCHIP))] + sdefs,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_bsend.cpp"),
                    core_ranges=crs([ac.local]),
                    compile_time_args=ct_send,
                    runtime_args=ln_rt,
                    defines=[("WF_LOCAL", "1")] + sdefs,
                    config=dm(1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_bsend.cpp"),
                    core_ranges=crs([bcore[b[0]] for b in brs]),
                    compile_time_args=ct_send,
                    runtime_args=br_rt,
                    defines=FAB_DEFS + sdefs,
                    config=dm(1),
                ),
            ]
            ki_relay, ki_br = len(gprog.kernels), len(gprog.kernels) + 3
            cbs = list(gprog.cbs) + [cb(0, cfg.w_dt, ac.SCB * ac.CHT, crs(scores))]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(gprog.semaphores), cbs=cbs)
            # fabric connection args: relay on link 1 of every neighbour, branch senders on link 0 of theirs
            rt = p.kernels[ki_relay].runtime_args[ac.relay[0]][ac.relay[1]]
            for nb in nbrs:
                rt.extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, rc),
                        dst_fabric_node_id=node(mesh, nb),
                        link_idx=1,
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(*ac.relay),
                    )
                )
            for o in range(NCHIP):
                orc = (o // Cn, o % Cn)
                if o == me:
                    sc = ac.local
                elif orc[0] == r:
                    sc = bcore["E"] if orc[1] < c else bcore["W"]
                else:
                    sc = bcore["NS"]
                sx, sy = noc(mesh, *sc)
                rt.extend([o | sx << 8 | sy << 16])
            for name, nb, _, _ in brs:
                x, y = bcore[name]
                p.kernels[ki_br].runtime_args[x][y].extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, rc),
                        dst_fabric_node_id=node(mesh, nb),
                        link_idx=0,
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(x, y),
                    )
                )
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    return mpd, io + [icred, sst], st, sst
