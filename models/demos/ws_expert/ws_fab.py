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

from dataclasses import dataclass, field

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


def slice_layout(cfg: GCfg, CHT, wg, wu, wd, ks=False):
    """The senders' DRAM copy of the weights: per expert e and group core j, the core's slice (gate/up pair j
    [KT][2], then down columns [DCOLS j, DCOLS (j + 1)): [KTD][DCOLS]) in CHT-tile chunks q = (e GC + j) NCH + u, chunk q
    stored as pages (q // 8) CHT 8 + q % 8 + 8 k (k < CHT): CHT consecutive tiles of one DRAM bank (one read a chunk).
    ks (the lane's K-split gate/up, LCfg.ks): core j = 10 c + r's gate/up part is K tiles [KSL r, KSL (r + 1)) x
    interleaved columns [20 c, 20 c + 20) ([KSL][20], KSL = KT / 10) instead of pair j over all K.
    Returns a [pages * 32, 32] torch tensor (tile layout rows)."""
    E, KT, KTD, DC, P, HT = wg.shape[0], cfg.KT, cfg.PAIRS, cfg.DCOLS, cfg.PAIRS, cfg.HT
    from models.demos.ws_expert.ws_expert import interleave_gu

    tiles = []
    for e in range(E):
        gu = interleave_gu(wg[e], wu[e]).reshape(KT, 32, 2 * P, 32).permute(0, 2, 1, 3)  # [KT, 2P, 32, 32]
        dn = wd[e].reshape(KTD, 32, HT, 32).permute(0, 2, 1, 3)  # [KTD, HT, 32, 32]
        for j in range(cfg.GC):
            if ks:
                ksl, half = KT // 10, cfg.GC // 2
                c, r = j // half, j % half
                tiles.append(gu[ksl * r : ksl * (r + 1), 2 * half * c : 2 * half * (c + 1)].reshape(-1, 32, 32))
            else:
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
# every chip: one core a fabric branch (E-only, W-only, N and S trunks with E/W teeth: 2D multicast) plus a local
# lander (NoC). Each chip's relay forwards freed slots to the owner-side senders that cover the chip.
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ACfg:
    CHT: int = 24
    SCB: int = 8
    RD: int = 4
    NQ: int = 0  # credit queue entries a link-0 branch sender; 0: room for every credit a chip sends (no wrap)
    relay: tuple = (10, 9)
    local: tuple = (9, 9)  # local lander
    branch: dict = field(  # branch -> (link 0 core, link 1 core)
        default_factory=lambda: {
            "E": ((10, 8), (10, 7)),
            "W": ((9, 8), (9, 7)),
            "N": ((10, 6), (9, 6)),
            "S": ((10, 5), (9, 5)),
        }
    )
    torus: bool = False  # both axes wrap (FABRIC_2D_TORUS_XY): the branches use the wrap links
    own_slices: bool = False  # ws_t holds only this chip's experts (e % NCHIP == chip), in expert order


def branches(mesh_shape, rc, torus=False):
    """[(name, first-hop neighbour, (e, w, n, s), covered chips)] of chip rc's multicast branches: E-only and W-only
    along its row, N and S trunks with E/W teeth on every row they reach. On a torus the extents follow the fabric's
    shortest routes, whose tie (the half-way chip of an even ring) goes forward (E, S): E covers C // 2 columns,
    W (C - 1) // 2, S R // 2 rows, N (R - 1) // 2; a multicast range must match those routes."""
    R, Cn = mesh_shape
    r, c = rc
    if torus:
        e, w, n, s = Cn // 2, (Cn - 1) // 2, (R - 1) // 2, R // 2
    else:
        e, w, n, s = Cn - 1 - c, c, r, R - 1 - r
    col = lambda k: (c + k) % Cn
    row = lambda k: (r + k) % R
    cols = [col(k) for k in range(-w, e + 1)]
    out = []
    if e:
        out.append(("E", (r, col(1)), (e, 0, 0, 0), [(r, col(k)) for k in range(1, e + 1)]))
    if w:
        out.append(("W", (r, col(-1)), (0, w, 0, 0), [(r, col(-k)) for k in range(1, w + 1)]))
    if n:
        out.append(("N", (row(-1), c), (e, w, n, 0), [(row(-k), cc) for k in range(1, n + 1) for cc in cols]))
    if s:
        out.append(("S", (row(1), c), (e, w, 0, s), [(row(k), cc) for k in range(1, s + 1) for cc in cols]))
    return out


def build_all(mesh, cfg: GCfg, ac: ACfg, x_t, idx_t, counts_t, ws_t, out_t):
    R, Cn = tuple(mesh.shape)
    NCHIP = R * Cn
    NC = cfg.G * cfg.GC
    chip = lambda rc: rc[0] * Cn + rc[1]
    rc_of = lambda o: (o // Cn, o % Cn)
    bcore = ac.branch  # name -> (link 0 core, link 1 core)
    all_b = [x for pair in bcore.values() for x in pair]
    send_cores = [ac.local] + all_b
    # every freed slot of a remote owner's expert is one queue entry: at most NC cores x (experts a group)
    NQ = ac.NQ or NC * -(-cfg.nexp // cfg.G) + 8
    icred = sharded_l1(mesh, [ac.relay] + send_cores, [NCHIP * NC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    sst = sharded_l1(mesh, send_cores, [1, 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    cq = sharded_l1(mesh, [b[0] for b in bcore.values()], [NQ, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    from models.demos.ws_expert.ws_group import group_cores

    gcores = [c for g in range(cfg.G) for c in group_cores(cfg, g)]
    assert not set([ac.relay] + send_cores) & set(gcores)
    gprog, io, st, ex = build(mesh, cfg, x_t, idx_t, counts_t, ws_t, ws_t, out_t, fab=dict(relay=ac.relay, icred=icred))
    tword = lambda c: c | noc(mesh, *gcores[c])[0] << 8 | noc(mesh, *gcores[c])[1] << 16
    tl_all = [tword(c) for c in range(NC)]
    tl_link = [[tword(c) for c in range(NC) if c % 2 == l] for l in range(2)]
    KT, KTD, DC = cfg.KT, cfg.PAIRS, cfg.DCOLS
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    xy = lambda core: noc(mesh, *core)[0] | noc(mesh, *core)[1] << 8
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    DIR = {"E": 0, "W": 1, "N": 2, "S": 3}  # eth_chan_directions
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            rc, me = (r, c), chip((r, c))
            brs = branches((R, Cn), rc, ac.torus)
            # relay: link-0 branch core a direction, local lander, then per owner chip and link the covering core
            qxy = [0, 0, 0, 0]
            for name, nb, rng, _ in brs:
                qxy[DIR[name]] = xy(bcore[name][0])
            owner_map = []
            for o in range(NCHIP):
                if o == me:
                    pair = (ac.local, ac.local)
                else:
                    pair = bcore[next(nm for nm, _, _, cov in branches((R, Cn), rc_of(o), ac.torus) if rc in cov)]
                owner_map += [xy(pair[0]), xy(pair[1])]
            rrt = ttnn.RuntimeArgs()
            rrt[ac.relay[0]][ac.relay[1]] = [me] + qxy + [xy(ac.local)] + owner_map
            relay = ttnn.KernelDescriptor(
                kernel_source=kpath("wf_relay_all.cpp"),
                core_ranges=crs([ac.relay]),
                compile_time_args=[NC, cfg.GC, cfg.G, cfg.nexp, icred.buffer_address(), NCHIP, mesh_id]
                + [cq.buffer_address(), NQ],
                runtime_args=rrt,
                defines=FAB_DEFS,
                config=dm(0),
            )
            rd_rt, ln_rt, b0_rt, b1_rt = (ttnn.RuntimeArgs() for _ in range(4))
            lx, ly = ac.local
            rd_rt[lx][ly] = [ws_t.buffer_address(), me, NC] + tl_all
            ln_rt[lx][ly] = [mesh_id, me, 0, 0, 0, 0, 1 << me, NC] + tl_all
            bc = [[], []]
            for name, nb, (e, w, n, s), cov in brs:
                for l in range(2):
                    x, y = bcore[name][l]
                    bc[l].append((x, y))
                    rd_rt[x][y] = [ws_t.buffer_address(), me, len(tl_link[l])] + tl_link[l]
                    (b0_rt, b1_rt)[l][x][y] = [mesh_id, me, e, w, n, s, sum(1 << chip(k) for k in cov)]
                    (b0_rt, b1_rt)[l][x][y].extend([len(tl_link[l])] + tl_link[l])
            scores = [ac.local] + bc[0] + bc[1]
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
                    defines=[("RD_DEPTH", str(ac.RD)), ("SCB_CHUNKS", str(ac.SCB)), ("NOWN", str(NCHIP))]
                    + ([("WS_OWN_SLICES", "1")] if ac.own_slices else [])
                    + sdefs,
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
                    core_ranges=crs(bc[0]),
                    compile_time_args=ct_send,
                    runtime_args=b0_rt,
                    defines=FAB_DEFS + sdefs + [("WF_CREDQ", str(cq.buffer_address())), ("WF_NQ", str(NQ))],
                    config=dm(1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_bsend.cpp"),
                    core_ranges=crs(bc[1]),
                    compile_time_args=ct_send,
                    runtime_args=b1_rt,
                    defines=FAB_DEFS + sdefs,
                    config=dm(1),
                ),
            ]
            kb = len(gprog.kernels) + 3
            cbs = list(gprog.cbs) + [cb(0, cfg.w_dt, ac.SCB * ac.CHT, crs(scores))]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(gprog.semaphores), cbs=cbs)
            for name, nb, _, _ in brs:
                for l in range(2):
                    x, y = bcore[name][l]
                    p.kernels[kb + l].runtime_args[x][y].extend(
                        ttnn.setup_fabric_connection(
                            mesh_device=mesh,
                            src_fabric_node_id=node(mesh, rc),
                            dst_fabric_node_id=node(mesh, nb),
                            link_idx=l,
                            program_descriptor=p,
                            worker_core=ttnn.CoreCoord(x, y),
                        )
                    )
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    return mpd, io + [icred, sst, cq], st, sst
