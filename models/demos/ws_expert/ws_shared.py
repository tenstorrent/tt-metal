# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared expert as one single-chip program (kernels wsh_*.cpp): y = (silu(x W1) * (x W3)) W2 for every one of the
chip's T tokens, written row-major into row t S + KSH of the layer's result buffer (the end pass adds it with weight
1). Weights replicated, resident in L1 (bfp4 by default); x row-major bf16 [TX, H] from DRAM (the stream's x).

Layout (the column clusters of ws_cluster / ws_dyn, sized for the shared inner dim): NCL clusters of GC cores, each a
rectangle of whole logical columns contiguous in NoC coordinates; cluster core j owns gate/up pairs NPAIR j ..
(NPAIR = I / 32 / GC, all of a pair's K) and down columns [C0, C0 + DC) (DC = 2 or 3: 120 columns over 46 cores).
Row tiles go round-robin to the clusters (no row is split, so no partials cross clusters). Per row tile every core
of the cluster computes its pairs' gate/up over the whole x row, sends its h tiles to the cluster's hub and computes
the previous row's down for its columns over the whole h row (K = I / 32). A core of the rectangle without weights,
the hub (at the rectangle's NOC1 multicast start corner), is the cluster's only multicaster: it reads each row tile's
32 whole token rows from DRAM, tilizes the 120 tiles (double-buffered, so the next tilize overlaps the multicasts),
multicasts them in BK-tile blocks into every core's x ring in stream order, and multicasts each h row once the
cores' h tiles landed in its own h ring. (Every core multicasting its own h tiles - 46 sources into one rectangle -
ran at a tenth of the speed; more x feeders (NF) add multicast sources and were slower too.)
NoCs: every DRAM read on NOC0 (the BRISCs: weight preload, x gather); the multicasts, the in-cluster unicasts and
the posted row writes on NOC1 (the NCRISCs), so nothing from outside a cluster lands on its multicast NoC and no
acknowledged DRAM write shares a NoC with DRAM reads."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import interleave_gu, noc, noc_rects, sharded_l1


@dataclass
class SCfg:
    T: int = 31250
    H: int = 3840
    IS: int = 2944
    GC: int = 46  # cores a cluster
    NCL: int = 2
    BK: int = 24  # x block (K-tiles): larger blocks, fewer multicasts
    NX: int = 9  # x ring blocks (<= 2 KT / BK - 1: the h ring's no-credit bound)
    NS: int = 3  # h ring rows
    NF: int = 1  # x feeders a cluster (the hub, then the rectangle's other cores without weights; more feeders put
    # more multicast sources into the rectangle: slower)
    XRM: int = 1  # row tiles of a feeder's row-major x staging (CB 7)
    XST: int = 2  # row tiles of a feeder's tilized x staging (CB 8): the next tilize overlaps the multicasts
    S: int = 13  # result rows a token
    KSH: int = 12  # this op's row of a token
    glu: int = 3
    w_dt: object = ttnn.bfloat4_b
    x_dt: object = ttnn.bfloat8_b
    h_dt: object = ttnn.bfloat8_b  # (bf16: twice the h multicast bytes, about 1.2x the time)
    out_dt: object = ttnn.bfloat16
    fidelity: object = ttnn.MathFidelity.LoFi
    layout: str = "cols"  # "cols": rectangles of whole logical columns; "rows": of whole rows (4 a cluster, from row0)
    row0: int = 0
    avoid: tuple = ()  # logical cores the clusters leave out (another program's, running at the same time)

    KT = property(lambda s: s.H // 32)
    HT = property(lambda s: s.H // 32)
    PAIRS = property(lambda s: s.IS // 32)
    NPAIR = property(lambda s: s.PAIRS // s.GC)
    KTD = property(lambda s: s.IS // 32)
    R = property(lambda s: -(-s.T // 32))  # row tiles
    NB = property(lambda s: s.KT // s.BK)  # x blocks a row


def clusters(mesh, cfg: SCfg):
    """NCL rectangles of whole logical columns, contiguous in NoC coordinates, each with more than GC cores: (rectangle
    cores, the GC active ones in column-major order, the x feeders: the hub (the rectangle's NOC1 multicast start
    corner) then the rest of the rectangle's cores)."""
    g = mesh.compute_with_storage_grid_size()
    if cfg.layout == "rows":
        # (a rectangle of whole rows may span NoC sub-rectangles - Blackhole's Tensix columns jump over the DRAM
        # columns: the hub, the core with the largest NoC (x, y), starts its own; the others lie at smaller x, so
        # its NOC1 multicasts reach them without wrapping)
        h = -(-(cfg.GC + 1) // g.x)
        bad = {yy for _, yy in cfg.avoid}  # (bands of whole rows: a row with an avoided core is skipped)
        starts, y0 = [], cfg.row0
        while y0 + h <= g.y and len(starts) < cfg.NCL:
            if any(yy in bad for yy in range(y0, y0 + h)):
                y0 += 1
                continue
            starts.append(y0)
            y0 += h
        assert len(starts) == cfg.NCL, f"{len(starts)} bands of {h} rows clear of {sorted(cfg.avoid)}"
        out = []
        for ys in starts:
            rect = [(x, y) for y in range(ys, ys + h) for x in range(g.x)]
            hub = max(rect, key=lambda cc: noc(mesh, *cc))
            act = [cc for cc in rect if cc != hub][: cfg.GC]
            out.append((rect, act, ([hub] + [cc for cc in rect if cc != hub and cc not in act])[: cfg.NF]))
        return out
    w = -(-(cfg.GC + 1) // g.y)
    out, x = [], 0
    while x + w <= g.x and len(out) < cfg.NCL:
        rect = [(xx, y) for xx in range(x, x + w) for y in range(g.y)]
        if len(noc_rects(mesh, rect)) == 1 and not set(rect) & set(cfg.avoid):
            hub = max(rect, key=lambda c: noc(mesh, *c))
            act = [c for c in rect if c != hub][: cfg.GC]
            out.append((rect, act, ([hub] + [c for c in rect if c != hub and c not in act])[: cfg.NF]))
            x += w
        else:
            x += 1
    assert len(out) == cfg.NCL, f"{len(out)} clusters of {w} columns fit the {g.x} x {g.y} grid"
    return out


def shared_weights(mesh, cfg: SCfg, w1, w3, w2):
    """Device weights (replicated): the gate/up matrix [H, 2 I] (32-column gate / up tiles interleaved) and W2
    [I, H]."""
    t = lambda m: ttnn.from_torch(
        m, dtype=cfg.w_dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return t(interleave_gu(w1.float(), w3.float())), t(w2.float())


def build_shared(mesh, cfg: SCfg, x_t, w13_t, w2_t, res_t):
    """generic_op mesh program of the shared expert; returns (mesh program, io tensors, stamps, active cores). res_t
    has NUM_DRAM_BANKS spare rows from T S (the posted row writes' fence)."""
    pt = shared_parts(mesh, cfg, x_t, w13_t, w2_t, res_t)
    p = ttnn.ProgramDescriptor(kernels=pt["kernels"], semaphores=pt["sems"], cbs=pt["cbs"])
    R, Cn = tuple(mesh.shape)
    mpd = ttnn.MeshProgramDescriptor()
    mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, Cn - 1))] = p
    return mpd, pt["io"], pt["stamps"], pt["active"]


def shared_parts(mesh, cfg: SCfg, x_t, w13_t, w2_t, res_t, fold=None):
    """The shared expert's kernels, CBs, semaphores, io tensors, stamps, active cores and clusters (a dict). fold (ws_fold):
    the cluster cores send their rows into the end cores' L1 rings instead of res_t (wsh_reader.cpp WSH_FOLD): a dict of
    ends (per cluster its end cores, logical), ring, flags (end cores' L1 addresses), fcred, fscr (L1 tensors on the
    active cores)."""
    assert cfg.PAIRS % cfg.GC == 0 and cfg.KT % cfg.BK == 0 and cfg.NX <= 2 * cfg.NB - 1 and cfg.NS == 3
    KT, KTD, HT, BK, NX, NS, GC, NPAIR = cfg.KT, cfg.KTD, cfg.HT, cfg.BK, cfg.NX, cfg.NS, cfg.GC, cfg.NPAIR
    GW = 2 * NPAIR
    cl = clusters(mesh, cfg)
    every = [c for rect, _, _ in cl for c in rect]
    active = [c for _, act, _ in cl for c in act]
    feeders = [f for _, _, fs in cl for f in fs]
    xtile, htile = TILE_BYTES[cfg.x_dt], TILE_BYTES[cfg.h_dt]
    w16 = lambda n: sharded_l1(mesh, every, [n, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xring = sharded_l1(mesh, every, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    hring = sharded_l1(mesh, every, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    xflags, xprog, hflags, hcnt = w16(NX), w16(GC), w16(NS), w16(NS)
    dcmax = -(-HT // GC)
    ysh = sharded_l1(mesh, every, [2 * dcmax * 32, 32], cfg.out_dt, ttnn.TILE_LAYOUT)  # (row writes: 64 B aligned)
    st = stamps_tensor(mesh, active)
    hst = stamps_tensor(mesh, feeders)
    # down columns: the first HT % GC cores of a cluster take one more
    base, extra = HT // GC, HT % GC
    dc_of = lambda j: base + (1 if j < extra else 0)
    c0_of = lambda j: j * base + min(j, extra)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_0 if risc == 0 else ttnn.NOC.NOC_1,
    )
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=cfg.fidelity, fp32_dest_acc_en=True)
    cdefs = [("WC_GLU", str(cfg.glu))] + ([("WX_X_FMT_DIFFERS", "1")] if cfg.x_dt != cfg.h_dt else [])
    cdefs += [  # (expert_activation.hpp, which the GLU builds on)
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    geo = []  # per cluster: rows, blocks, NOC1 multicast (sub-)rectangles (the hub's first) with their cores, feeders
    for c, (rect, act, fs) in enumerate(cl):
        rows = len(range(c, cfg.R, cfg.NCL))
        hx, hy = noc(mesh, *fs[0])
        rs = sorted(noc_rects(mesh, rect), key=lambda r: not (r[0] <= hx <= r[2] and r[1] <= hy <= r[3]))
        assert (rs[0][2], rs[0][3]) == (hx, hy) and all(r[2] < hx for r in rs[1:]), "the hub starts its sub-rectangle"
        sub = [[r[2], r[3], r[0], r[1], r[4]] for r in rs]
        fxy = [noc(mesh, *f)[0] | noc(mesh, *f)[1] << 8 for f in fs]
        geo.append((rows, rows * cfg.NB, sub, len(rect), fxy))
    kernels = []
    # (the start barrier: the cluster cores and the hubs count in at the master - the first hub; the fold's end core 0)
    xy = lambda c: noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8
    master = fold["master"] if fold else xy(feeders[0])
    members = [xy(c) for c in active + feeders]
    # the x feeders: gather (BRISC, NOC0), tilize (compute), multicasts (NCRISC, NOC1; the hub also the h rows)
    h_rt = [ttnn.RuntimeArgs() for _ in range(3)]
    for c, (_, _, fs) in enumerate(cl):
        rows, nblk, r1, nrec, _ = geo[c]
        NF = len(fs)
        for f, (x, y) in enumerate(fs):
            own = len(range(f, rows, NF))
            h_rt[0][x][y] = [x_t.buffer_address(), c, f, NF, rows, 0, 0, 0, hst.buffer_address()]
            h_rt[1][x][y] = [own, 0, 0, 0, 0, 0, 0, 0, hst.buffer_address()]
            first = not fold and (x, y) == feeders[0]
            h_rt[2][x][y] = [f, NF, rows] + r1[0] + [hst.buffer_address(), int(f == 0), len(r1)] + sum(r1[1:], []) + [
                master,
                int(first),
            ] + ([len(members)] + members if first else [])
    fset = crs(feeders)
    kernels += [
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsh_hub.cpp"),
            core_ranges=fset,
            compile_time_args=[KT, cfg.NCL, cfg.H * 2] + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args()),
            defines=[("WSH_ROLE", "0")],
            runtime_args=h_rt[0],
            config=dm(0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsh_hub.cpp"),
            core_ranges=fset,
            compile_time_args=[KT],
            defines=[("WSH_ROLE", "1")],
            runtime_args=h_rt[1],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsh_hub.cpp"),
            core_ranges=fset,
            compile_time_args=[BK, KTD, NX, NS, GC, xring.buffer_address(), xflags.buffer_address()]
            + [xprog.buffer_address(), hring.buffer_address(), hflags.buffer_address(), hcnt.buffer_address()]
            + [xtile, htile, KT],
            defines=[("WSH_ROLE", "2")],
            runtime_args=h_rt[2],
            config=dm(1),
        ),
    ]
    # the cluster cores, one kernel set per down width
    for DC in sorted({dc_of(j) for j in range(GC)}):
        wr_rt, rd_rt, cp_rt, cores = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), []
        for c, (rect, act, fs) in enumerate(cl):
            rows, nblk, _, _, fxy = geo[c]
            for j, (x, y) in enumerate(act):
                if dc_of(j) != DC:
                    continue
                cores.append((x, y))
                wr_rt[x][y] = [w13_t.buffer_address(), w2_t.buffer_address(), j, c0_of(j)]
                rd_rt[x][y] = [j, rows, nblk, c0_of(j), c, res_t.buffer_address(), cfg.T, len(fxy)] + fxy + (
                    [xy(e) for e in fold["ends"][c]] if fold else []
                ) + [master]
                cp_rt[x][y] = [st.buffer_address(), rows]
        cs = crs(cores)
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=kpath("wsh_writer.cpp"),
                core_ranges=cs,
                compile_time_args=[KT, GW, 2 * cfg.PAIRS, KTD, DC, HT]
                + list(ttnn.TensorAccessorArgs(w13_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(w2_t).get_compile_time_args()),
                runtime_args=wr_rt,
                config=dm(0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("wsh_reader.cpp"),
                core_ranges=cs,
                compile_time_args=[NPAIR, BK, KTD, DC, NS, NX, xflags.buffer_address(), xprog.buffer_address()]
                + [hring.buffer_address(), hflags.buffer_address(), hcnt.buffer_address(), htile]
                + [cfg.S, cfg.KSH, cfg.H * 2, cfg.NCL, KT]
                + list(ttnn.TensorAccessorArgs(res_t).get_compile_time_args())
                + (
                    [len(fold["ends"][0]), fold["ring"], fold["flags"], fold["fcred"].buffer_address()]
                    + [fold["fscr"].buffer_address(), GC]
                    if fold
                    else []
                ),
                defines=[("WSH_FOLD", "1")] if fold else [],
                runtime_args=rd_rt,
                config=dm(1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("wsh_compute.cpp"),
                core_ranges=cs,
                compile_time_args=[KT, NPAIR, BK, KTD, DC],
                named_compile_time_args=[("expert_activation", 0)],
                defines=cdefs,
                runtime_args=cp_rt,
                config=ccfg,
            ),
        ]
    aset = crs(active)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, xring),
        cb(1, cfg.w_dt, KT * GW, aset),
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        cb(3, cfg.w_dt, KTD * dcmax, aset),
        cb(7, ttnn.bfloat16, cfg.XRM * KT, fset),  # (row tiles of 32 token rows)
        cb(8, cfg.x_dt, cfg.XST * KT, fset),
        cb(16, cfg.h_dt, 2 * NPAIR, aset),
        ttnn.cb_descriptor_from_sharded_tensor(17, ysh),
    ]
    # (2, 3: the start barrier; the fold puts them on its end cores too)
    sems = [
        ttnn.SemaphoreDescriptor(id=s, core_ranges=crs(active + feeders), initial_value=0)
        for s in (range(2) if fold else range(4))
    ]
    io = [x_t, w13_t, w2_t, xring, hring, xflags, xprog, hflags, hcnt, ysh, st, hst]
    io += [fold["fcred"], fold["fscr"]] if fold else []
    return dict(
        kernels=kernels,
        cbs=cbs,
        sems=sems,
        io=io + [res_t],
        stamps=(st, hst),
        active=active,
        feeders=feeders,
        clusters=cl,
    )


def shared_span_us(sts, ci, clk):
    """The shared expert's device time on chip ci: earliest compute start to latest compute end over its cores."""
    st = sts[0]
    s = ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF
    s = s.reshape(-1, 8 * 64)
    rel = lambda v: ((v - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return (int(rel(s[:, 63]).max()) - int(rel(s[:, 0]).min())) / clk


def shared_waits_k(sts, ci):
    """Mean over the cluster cores (K cycles): x waits, h row waits; and per hub role its busy fraction (1 - waits /
    its span) with the waits: gather (read waits, space waits), tilize (input, output space), multicasts (x multicast
    time, h multicast time, credit waits, arrival waits)."""
    s = (ttnn.to_torch(ttnn.get_device_tensors(sts[0])[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 8 * 64).double()
    out = {n: round(float(s[:, 41 + i].mean()) / 1000, 1) for i, n in enumerate(("x_wait", "h_wait"))}
    h = (ttnn.to_torch(ttnn.get_device_tensors(sts[1])[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 8 * 64).double()
    span = lambda a, b: (h[:, b] - h[:, a]).remainder(2**32)
    m = lambda v: round(float(v.mean()), 3)
    g, t, x = span(0, 3), span(8, 11), span(16, 21)
    out["hub_gather"] = dict(busy=m(1 - (h[:, 1] + h[:, 2]) / g), reads=m(h[:, 1] / g), space=m(h[:, 2] / g))
    out["hub_tilize"] = dict(busy=m(1 - (h[:, 9] + h[:, 10]) / t), input=m(h[:, 9] / t), space=m(h[:, 10] / t))
    out["hub_mcast"] = dict(
        x_mcast=m(h[:, 17] / x), h_mcast=m(h[:, 18] / x), credit_wait=m(h[:, 19] / x), arrival_wait=m(h[:, 20] / x)
    )
    return out
