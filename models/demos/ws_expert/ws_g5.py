# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert K-split groups of 5 x 5 cores (one chip, injector-landed weights; the ws_ks harness).

Group = 5 columns c x 5 rows r, core j = 5 c + r. Expert e runs on group e % G.
- Gate/up: K split over the rows, N over the columns. Rows 0-3 hold KSL 25 K tiles each, row 4 holds 20
  (120 = 4 x 25 + 20); column c holds pairs 4 c .. 4 c + 3 (8 gate/up tiles a row). Core (c, r) computes the partials
  of its column's 4 pairs for every x row (2 blocks of 1 x 4).
- Pair owners: core (c, o), o < 4, owns pair 4 c + o: the column's 5 partials of that pair are reduce-scattered to it
  (rows below on NOC0 from the BRISC, rows above on NOC1 from the NCRISC), it folds them in fp32 DEST, applies SwiGLU
  and produces h tile 4 c + o. Row 4 owns no pair.
- Down: owners 4 output columns each ([4 (4 c + o), +4)), row 4 cores 8 each ([80 + 8 c, +8)): every core does 320
  tile products a row including the fold (owner: 25 x 8 + 20 x 4 + ~40 fold; row 4: 20 x 8 + 20 x 8).
- x: row r's slice x[:, K_r] is needed by the 5 cores of the row; core (c, r) gathers and tilizes rows q with
  q % 5 == c and the BRISC unicasts them into the 5 cores' x rings (one flag a slot; credits per consumer).
- h: owners send their tile to the column leader of the row (core (c, q % 4)), which multicasts the column's 4 tiles
  to the group (even columns from the BRISC on NOC0, odd from the NCRISC on NOC1); every core waits for 5 flags.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1
from models.demos.ws_expert.ws_group import GCfg

NR, NCOL = 5, 5  # rows, columns of a group
OWN_ROWS = 4  # rows 0..3 own pairs


@dataclass
class G5Cfg(GCfg):
    GC: int = 25
    G: int = 2
    NXR: int = 8  # x ring rows
    D: int = 4  # landing slots (rows of 5 contributions)
    PD: int = 2  # partial ring rows
    FL: int = 2  # fold lag (rows)
    TL: int = 9  # tilize look-ahead (rows; own rows are 1 in 5, so it must reach past the next one: TL 3 61.7%)
    XRM: int = 2  # gathered row-major staging (rows)
    NS: int = 6
    dn_lag: int = 3
    p_dt: object = ttnn.bfloat16
    NINJ: int = 25
    P: int = 2
    origins: tuple = ((0, 0), (0, 5), (6, 0), (6, 5))  # top-left logical core of group g (must be NoC contiguous)

    PC = property(lambda s: s.PAIRS // NCOL)  # pairs a column (4)
    NW = property(lambda s: 2 * s.PAIRS // NCOL)  # partial tiles a row (8)


def row_geom(cfg, r):
    """(K offset, KSL, down columns) of row r."""
    big = 25
    small = cfg.KT - OWN_ROWS * big
    return (big * r, big, 4) if r < OWN_ROWS else (OWN_ROWS * big, small, 8)


def core_geom(cfg, j):
    """(c, r, K offset, KSL, down column start, down columns, owner) of core j = 5 c + r."""
    c, r = j // NR, j % NR
    k0, ksl, dc = row_geom(cfg, r)
    ds = (cfg.PC * c + r) * dc if r < OWN_ROWS else OWN_ROWS * 4 * NCOL + dc * c
    return c, r, k0, ksl, ds, dc, r < OWN_ROWS


def g5_cores(cfg, g):
    x0, y0 = cfg.origins[g]
    return [(x0 + c, y0 + r) for c in range(NCOL) for r in range(NR)]  # core j = 5 c + r


def inj_cores(cfg, groups):
    used = {c for gr in groups for c in gr}
    free = [(x, y) for x in range(11) for y in range(10) if (x, y) not in used]
    assert len(free) >= cfg.NINJ
    return free[: cfg.NINJ]


def build(device, cfg: G5Cfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t):
    assert cfg.GC == NR * NCOL and cfg.PAIRS == OWN_ROWS * NCOL and cfg.KT == 120 and cfg.HT == 120
    assert cfg.NINJ == cfg.GC, "one injector a group core index (NT 1)"
    assert cfg.wsrc == "l1"
    assert cfg.NS >= cfg.dn_lag + 1
    groups = [g5_cores(cfg, g) for g in range(cfg.G)]
    C = [c for gr in groups for c in gr]
    INJ = inj_cores(cfg, groups)
    KTD, NS, NW, LT = cfg.PAIRS, cfg.NS, cfg.NW, 2 * NR
    DB = 4
    WGU_MAX, WDN_MAX = 25 * NW, KTD * 8
    wtile, htile, xtile, ptile = (TILE_BYTES[d] for d in (cfg.w_dt, cfg.h_dt, cfg.x_dt, cfg.p_dt))
    U, RM = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    w16 = lambda n, cores=C: sharded_l1(device, cores, [n, 4], U, RM, torch.int32)
    icred = w16(cfg.G, INJ)
    warr = w16(2)
    wgu_ring = sharded_l1(device, C, [2 * WGU_MAX * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    wd_ring = sharded_l1(device, C, [2 * WDN_MAX * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    hring = sharded_l1(device, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = w16(NS * NCOL)
    xring = sharded_l1(device, C, [cfg.NXR * 25 * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags, xcred = w16(cfg.NXR), w16(NCOL)
    land = sharded_l1(device, C, [cfg.D * LT * 32, 32], cfg.p_dt, ttnn.TILE_LAYOUT)
    lflags, lcred = w16(cfg.D * NR), w16(NR)
    pring = sharded_l1(device, C, [cfg.PD * NW * 32, 32], cfg.p_dt, ttnn.TILE_LAYOUT)
    scr = w16(8)
    counts_l1 = sharded_l1(device, C, [1, 32 + max(4, cfg.nexp)], U, RM, torch.int32)
    idx_l1 = sharded_l1(device, C, [4, 32], U, RM, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base, idx_addr = a64(counts_l1.buffer_address()), a64(idx_l1.buffer_address())
    st = stamps_tensor(device, C)

    kernels = []
    cp_rt = {own: ttnn.RuntimeArgs() for own in (True, False)}
    for g, gr in enumerate(groups):
        nexp_g = len(range(g, cfg.nexp, cfg.G))
        rs = noc_rects(device, gr)
        assert len(rs) == 1, f"group {g} is not NoC contiguous: {rs}"
        r0 = list(rs[0][:4])
        r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
        cx = [noc(device, *gr[NR * c])[0] for c in range(NCOL)]
        ry = [noc(device, *gr[r])[1] for r in range(NR)]
        for own in (True, False):
            sub = [(j, xy) for j, xy in enumerate(gr) if core_geom(cfg, j)[6] == own]
            r_any = sub[0][0] % NR
            k0, ksl, dc = row_geom(cfg, r_any)
            gwr, grd = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for j, (x, y) in sub:
                c, r, k0j, kslj, ds, dcj, _ = core_geom(cfg, j)
                assert (kslj, dcj) == (ksl, dc)
                ix, iy = noc(device, *INJ[j])
                cp_rt[own][x][y] = [st.buffer_address(), j, g, nexp_g]
                gwr[x][y] = (
                    [j, g, nexp_g, x_t.buffer_address(), idx_t.buffer_address(), counts_t.buffer_address()]
                    + [
                        st.buffer_address(),
                        k0j,
                    ]
                    + cx
                    + ry
                )
                grd[x][y] = [j, g, nexp_g, ix, iy, g * 16, st.buffer_address(), out_t.buffer_address(), ds] + cx + ry
            common = [ksl, dc, int(own), cfg.NXR, xring.buffer_address(), xflags.buffer_address()]
            common += [xcred.buffer_address(), xtile, cfg.D, land.buffer_address(), lflags.buffer_address()]
            common += [lcred.buffer_address(), ptile, scr.buffer_address(), KTD, cfg.G, NS, hring.buffer_address()]
            common += [hflags.buffer_address(), htile, cnt_base]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("g5_writer.cpp"),
                    core_ranges=crs([xy for _, xy in sub]),
                    compile_time_args=common
                    + r0
                    + [cfg.RMAX, cfg.H * 2, idx_addr, cfg.nexp]
                    + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                    + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                    + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args()),
                    runtime_args=gwr,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            )
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("g5_reader.cpp"),
                    core_ranges=crs([xy for _, xy in sub]),
                    compile_time_args=common
                    + r_noc1
                    + [icred.buffer_address(), warr.buffer_address(), cfg.RMAX, cfg.HT, cfg.PD, pring.buffer_address()]
                    + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args()),
                    runtime_args=grd,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                    ),
                )
            )
    # injectors: injector i serves core j = i of every group (P weight sets preloaded in its L1)
    irr, iwr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(INJ):
        c, r, k0, ksl, ds, dc, _ = core_geom(cfg, i)
        irr[x][y] = [wgu_t.buffer_address(), wd_t.buffer_address(), k0, ksl, cfg.PC * 2 * c, ds, dc]
        iwr[x][y] = [ksl * NW, KTD * dc] + [v for gr in groups for v in noc(device, *gr[i])]
    WTOT_MAX = WGU_MAX + WDN_MAX
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("g5_inj_read.cpp"),
            core_ranges=crs(INJ),
            compile_time_args=[cfg.KT, KTD, cfg.HT, NW, 2 * cfg.PAIRS, cfg.P, WTOT_MAX]
            + list(ttnn.TensorAccessorArgs(wgu_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args()),
            runtime_args=irr,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
    )
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("g5_inj_write.cpp"),
            core_ranges=crs(INJ),
            compile_time_args=[cfg.nexp, cfg.G, cfg.P, WTOT_MAX, 40, WGU_MAX, WDN_MAX]
            + [wgu_ring.buffer_address(), wd_ring.buffer_address(), icred.buffer_address(), warr.buffer_address()],
            runtime_args=iwr,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        )
    )
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
        ("DN_LAG", str(cfg.dn_lag)),
    ]
    cbs = []
    for own in (True, False):
        cs = crs([xy for gr in groups for j, xy in enumerate(gr) if core_geom(cfg, j)[6] == own])
        _, ksl, dc = row_geom(cfg, 0 if own else OWN_ROWS)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("g5_compute.cpp"),
                core_ranges=cs,
                compile_time_args=[ksl, KTD, dc, DB, cnt_base, cfg.G, NW, cfg.FL, cfg.TL, int(own)],
                named_compile_time_args=[("expert_activation", 0)],
                defines=act_defs,
                runtime_args=cp_rt[own],
                config=ccfg,
            )
        )
        sh = ttnn.cb_descriptor_from_sharded_tensor
        cbs += [
            sh(0, xring, total_size=cfg.NXR * ksl * xtile, core_ranges=cs),
            sh(1, wgu_ring, total_size=2 * ksl * NW * wtile, core_ranges=cs),
            sh(3, wd_ring, total_size=2 * KTD * dc * wtile, core_ranges=cs),
            cb(7, ttnn.bfloat16, cfg.XRM * ksl, cs),
            cb(8, cfg.x_dt, (cfg.TL // NCOL + 3) * ksl, cs),  # (own rows are 1 in 5 of the look-ahead)
            cb(17, cfg.out_dt, 2 * dc, cs),
        ]
    cset = crs(C)
    cbs += [
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        ttnn.cb_descriptor_from_sharded_tensor(4, land),
        cb(16, cfg.h_dt, 2, cset),
        ttnn.cb_descriptor_from_sharded_tensor(18, pring),
        cb(0, cfg.w_dt, cfg.P * WTOT_MAX, crs(INJ)),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=crs(C + INJ), initial_value=0) for s in range(6)]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, idx_t, counts_t, wgu_t, wd_t, wgu_ring, wd_ring, hring, hflags, xring, xflags, xcred, land, lflags]
    io += [lcred, pring, scr, counts_l1, idx_l1, icred, warr, st, out_t]
    return prog, io, st, {}
