# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert with streamed weights: G groups of GC = 20 cores (two adjacent NoC-contiguous columns); expert e runs on
group e % G (each expert's weights land once per chip). Core j of a group owns gate/up pair j and down columns
[6 j, 6 j + 6); its weight slice (207 KB bfp4) lands in a 2-slot ring. On one chip the landing is done by NINJ = 20
injector cores (stand-in for the fabric all-gather: same ring addresses, arrival semaphore, credits), either from
weight sets preloaded in their L1 (wsrc "l1", fabric-like) or read from DRAM per expert (wsrc "dram": at 4 groups the
~130 GB/s of 576 B weight reads beside the x gather saturate DRAM).
x: gathered from DRAM by row index, tilized (bfp8 by default) and multicast inside the group; column 0 multicasts on
NOC0 (+x) and column 1 on NOC1 (-x), so no multicast wraps the row, one at a time a NoC in stream order (block s
waits for block s - 2 to land: concurrent linked multicasts starved single senders for up to 58K cycles).
h: each core sends its tile to its column's leader for the row; the leaders multicast 10 tiles each (column 0 from the
BRISC on NOC0, column 1 from the NCRISC on NOC1).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1

COL_PAIRS = [(0, 1), (2, 3), (4, 5), (7, 8), (9, 10)]  # logical column pairs, contiguous in NoC x


@dataclass
class GCfg:
    H: int = 3840
    I: int = 640
    nexp: int = 32
    G: int = 4
    GC: int = 20
    BK: int = 24
    NX: int = 15
    LOOK: int = 30
    MMAX: int = 2048
    T: int = 8192
    NINJ: int = 20
    inj_cols: tuple = (9, 10)  # logical columns of the injectors (outside the groups), 10 cores each
    CH: int = 120
    ICB: int = 2  # injector staging (chunks)
    XRM: int = 3  # gathered row-major staging (blocks)
    glu: int = 3
    dn_lag: int = 1
    NS: int = 4
    x_dt: object = ttnn.bfloat8_b  # (bf16 x: twice the multicast bytes - see WORKLOG)
    w_dt: object = ttnn.bfloat4_b
    h_dt: object = ttnn.bfloat16
    out_dt: object = ttnn.bfloat16
    wdefs: tuple = ()  # debug defines for the group BRISC (WG_NO_GATHER)
    wsrc: str = "l1"  # injectors: "l1" = P weight sets preloaded in L1, landed like the fabric would; "dram" = every
    # expert's slice read from DRAM in the steady state (DRAM-bound at G 4: ~130 GB/s of 576 B reads beside x)
    P: int = 2  # weight sets in "l1" mode (expert e uses set (e // G) % P: a group's consecutive experts differ)
    idefs: tuple = ()  # debug defines for the injector writer (WG_INJ_NOWRITE: only the arrival semaphores)
    xorder: int = 2  # x multicasts in stream order: block s waits for block s - xorder to land (0: off)
    trace0: int = -1  # debug: x block timeline from this stream index (40 blocks)

    KT = property(lambda s: s.H // 32)
    PAIRS = property(lambda s: s.I // 32)
    HT = property(lambda s: s.H // 32)
    DCOLS = property(lambda s: s.HT // s.GC)
    RMAX = property(lambda s: (s.MMAX + 31) // 32)


def group_cores(cfg, g):
    a, b = COL_PAIRS[g]
    return [(a, y) for y in range(10)] + [(b, y) for y in range(10)]  # core j = c * 10 + y


def build(device, cfg: GCfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t, fab=None):
    """fab: None (on-chip injectors) or dict(relay=logical core, icred=tensor holding NC credit words on it): the
    weights land over the fabric (ws_fab) and the cores' freed counts go to the relay core instead of injectors."""
    assert cfg.GC == 20 and cfg.PAIRS == cfg.GC and cfg.DCOLS * cfg.GC == cfg.HT
    groups = [group_cores(cfg, g) for g in range(cfg.G)]
    C = [c for gr in groups for c in gr]
    INJ = [] if fab else [(cfg.inj_cols[i // 10], i % 10) for i in range(cfg.NINJ)]
    assert not set(INJ) & set(C)
    cset = crs(C)
    KT, KTD, DC, BK, NX, NS = cfg.KT, cfg.PAIRS, cfg.DCOLS, cfg.BK, cfg.NX, cfg.NS
    DB = 3
    WGU, WDN = KT * 2, KTD * DC
    wtile, htile, xtile = TILE_BYTES[cfg.w_dt], TILE_BYTES[cfg.h_dt], TILE_BYTES[cfg.x_dt]
    NT = cfg.GC // cfg.NINJ
    # (first: the injectors' only L1 tensor, at the top of L1, so their CB 0 can be large in "l1" mode)
    if fab:
        icred = fab["icred"]
    else:
        icred = sharded_l1(device, INJ, [cfg.G * NT, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    # L1 tensors (same address on every core)
    warr = sharded_l1(device, C, [1, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)  # landed weight slices
    wgu_ring = sharded_l1(device, C, [2 * WGU * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    wd_ring = sharded_l1(device, C, [2 * WDN * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    hring = sharded_l1(device, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = sharded_l1(device, C, [NS * 2, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xring = sharded_l1(device, C, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags = sharded_l1(device, C, [NX, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xprog = sharded_l1(device, C, [cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    # DRAM -> L1 reads need the L1 address congruent to the DRAM one mod 64: 64-aligned landing spots inside these
    counts_l1 = sharded_l1(device, C, [1, 32 + max(4, cfg.nexp)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    idx_l1 = sharded_l1(device, C, [4, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base = a64(counts_l1.buffer_address())  # word 0 ready flag, words 16.. counts (64 B in)
    idx_addr = a64(idx_l1.buffer_address())
    nsets = cfg.P if cfg.wsrc == "l1" else cfg.nexp
    WTOT = WGU + WDN
    pre = cfg.wsrc == "l1"
    idefs_all = [(d, "1") for d in cfg.idefs] + ([("WG_INJ_P", str(cfg.P))] if pre else [])
    st = stamps_tensor(device, C)
    kernels = []
    cp_rt, rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    lag = [("DN_LAG", str(cfg.dn_lag))]
    for g, gr in enumerate(groups):
        nexp_g = len(range(g, cfg.nexp, cfg.G))
        rs = noc_rects(device, gr)
        assert len(rs) == 1, rs
        r0 = list(rs[0][:4])
        r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
        cx0, ny0 = noc(device, *gr[0])
        cx1, _ = noc(device, *gr[10])
        for j, (x, y) in enumerate(gr):
            assert noc(device, x, y) == ((cx0, cx1)[j // 10], ny0 + j % 10)
            if fab:
                ix, iy = noc(device, *fab["relay"])
                ioff = (g * cfg.GC + j) * 16
            else:
                ix, iy = noc(device, *INJ[j % cfg.NINJ])
                ioff = (g * NT + j // cfg.NINJ) * 16
            cp_rt[x][y] = [st.buffer_address(), j, g, nexp_g]
            wr_rt[x][y] = [j, g, nexp_g, cx0, cx1, ny0, out_t.buffer_address(), x_t.buffer_address()] + [
                idx_t.buffer_address(),
                counts_t.buffer_address(),
                st.buffer_address(),
            ]
            rd_rt[x][y] = [j, g, nexp_g, cx0, cx1, ny0, ix, iy, ioff, st.buffer_address()]
        gwr = ttnn.RuntimeArgs()
        grd = ttnn.RuntimeArgs()
        for x, y in gr:
            gwr[x][y] = wr_rt[x][y]
            grd[x][y] = rd_rt[x][y]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wg_writer.cpp"),
                core_ranges=crs(gr),
                compile_time_args=[KT, BK, KTD, DC, cfg.GC, cfg.G, NS, hring.buffer_address()]
                + [hflags.buffer_address()]
                + r0
                + [htile, cfg.HT, cfg.RMAX, cfg.H * 2, cnt_base, idx_addr, cfg.nexp]
                + [NX, xring.buffer_address(), xflags.buffer_address(), xprog.buffer_address(), xtile]
                + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args()),
                runtime_args=gwr,
                defines=[(d, "1") for d in cfg.wdefs]
                + [("WG_XORDER", str(cfg.xorder))]
                + ([("WG_TRACE0", str(cfg.trace0))] if cfg.trace0 >= 0 else []),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wg_reader.cpp"),
                core_ranges=crs(gr),
                compile_time_args=[KT, BK, KTD, DC, cfg.GC, cfg.G, NS, NX, hflags.buffer_address()]
                + [xring.buffer_address(), xflags.buffer_address(), xprog.buffer_address()]
                + r_noc1
                + [xtile, cnt_base, icred.buffer_address(), hring.buffer_address(), htile, warr.buffer_address()],
                runtime_args=grd,
                defines=[("WG_XORDER", str(cfg.xorder))]
                + ([("WG_TRACE0", str(cfg.trace0))] if cfg.trace0 >= 0 else []),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            )
        )
    if not fab:
        kernels += injector_kernels(
            device, cfg, groups, INJ, NT, wgu_t, wd_t, wgu_ring, wd_ring, icred, warr, idefs_all
        )
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    ccfg.unpack_to_dest_mode = utd
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("wg_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, BK, KTD, DC, DB, NX, cfg.LOOK, cnt_base, cfg.G, cfg.GC],
            named_compile_time_args=[("expert_activation", 0)],
            defines=act_defs
            + lag
            + [("WC_GLU", str(cfg.glu))]
            + ([("WX_X_FMT_DIFFERS", "1")] if cfg.x_dt != cfg.h_dt else [])
            + ([("WG_TRACE0", str(cfg.trace0))] if cfg.trace0 >= 0 else []),
            runtime_args=cp_rt,
            config=ccfg,
        )
    )
    XST = -(-cfg.LOOK // cfg.GC) + 4
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, xring),
        ttnn.cb_descriptor_from_sharded_tensor(1, wgu_ring),
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        ttnn.cb_descriptor_from_sharded_tensor(3, wd_ring),
        cb(7, ttnn.bfloat16, cfg.XRM * BK, cset),
        cb(8, cfg.x_dt, XST * BK, cset),
        cb(16, cfg.h_dt, 2, cset),
        cb(17, cfg.out_dt, 2 * DC, cset),
    ]
    if not fab:
        cbs.append(cb(0, cfg.w_dt, cfg.P * NT * WTOT if pre else cfg.ICB * cfg.CH, crs(INJ)))
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=crs(C + INJ), initial_value=0) for s in range(6)]
    # (semaphore 0 unused since the arrival counter became the warr tensor: its address is known to other chips)
    if fab:
        kernels += fab.get("kernels", [])  # (the caller's relay kernel: RT args filled in after, with fabric args)
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, idx_t, counts_t, wgu_t, wd_t, wgu_ring, wd_ring, hring, hflags, xring, xflags, xprog, counts_l1]
    io += [idx_l1, icred, warr, st, out_t]
    return prog, io, st, dict(wgu_ring=wgu_ring, wd_ring=wd_ring, warr=warr)


def injector_kernels(device, cfg, groups, INJ, NT, wgu_t, wd_t, wgu_ring, wd_ring, icred, warr, idefs_all):
    """On-chip weight landing (stand-in for the fabric): NINJ injector cores, reader (BRISC) + writer (NCRISC)."""
    KT, KTD, DC = cfg.KT, cfg.PAIRS, cfg.DCOLS
    iset = crs(INJ)
    kernels = []
    irr, iwr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(INJ):
        irr[x][y] = [i, wgu_t.buffer_address(), wd_t.buffer_address()]
        tg = []
        for g, gr in enumerate(groups):
            for t in range(NT):
                tg += list(noc(device, *gr[i + cfg.NINJ * t]))
        iwr[x][y] = [i] + tg
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("wg_inj_read.cpp"),
            core_ranges=iset,
            compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.PAIRS, cfg.HT, cfg.CH, NT, cfg.NINJ]
            + list(ttnn.TensorAccessorArgs(wgu_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args()),
            defines=idefs_all,
            runtime_args=irr,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
    )
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("wg_inj_write.cpp"),
            core_ranges=iset,
            compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.CH, NT, cfg.NINJ, cfg.G]
            + [wgu_ring.buffer_address(), wd_ring.buffer_address(), icred.buffer_address(), warr.buffer_address()],
            defines=idefs_all,
            runtime_args=iwr,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        )
    )
    return kernels
