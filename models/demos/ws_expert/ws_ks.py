# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert groups with a K-split gate/up (moe_fused_swiglu's K x N layout inside a 20-core group), one chip.

Group = 2 columns c x 10 rows r (core j = 10 c + r), expert e on group e % G. Gate/up: core (c, r) holds W_gu[K tiles
[KSL r, KSL (r + 1)), pairs 10 c .. 10 c + 9] (240 bfp4 tiles, as the 1 x 20 split's pair j over all K) and computes
partials of its column's 10 pairs for every row. x: core (c, r) needs x[:, K_r] only; it gathers and tilizes half of
it (KH = KSL / 2 K tiles) and sends it to itself and its partner (1 - c, r) - one hop, no chain. Partials: core
(c, r) owns pair 10 c + r; every core of the column sends it that pair's (gate, up) partial tiles (bf16) - the rows
below on NOC0 from the BRISC, the rows above on NOC1 from the NCRISC - and the owner folds the 10 contributions in
fp32 DEST, applies SwiGLU and produces h tile j, exactly as in the 1 x 20 groups (ws_group): h (column leaders,
multicasts), down and the weight landing (injectors) are unchanged.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1
from models.demos.ws_expert.ws_group import GCfg, group_cores, injector_kernels


@dataclass
class KCfg(GCfg):
    NXR: int = 6  # x ring rows (KSL tiles each)
    D: int = 4  # landing slots (rows of 10 contributions; even with fold2: a 2-row window must not wrap the ring)
    PD: int = 2  # partial ring rows (CB 18)
    FL: int = 1  # fold lag (rows of partials ahead of the fold)
    TL: int = 2  # tilize lookahead (rows)
    XRM: int = 4  # gathered row-major staging (row halves)
    p_dt: object = ttnn.bfloat16  # partials
    fold2: bool = True  # fold two rows at a time (one init a window; FL counts rows ahead of the pair)
    xalt: bool = True  # x: partners gather / tilize whole rows alternately (768 B reads) instead of halves of each row

    KSL = property(lambda s: s.KT // 10)


def build(device, cfg: KCfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t):
    assert cfg.GC == 20 and cfg.PAIRS == cfg.GC and cfg.DCOLS * cfg.GC == cfg.HT and cfg.KT % 20 == 0
    assert not cfg.fold2 or cfg.D % 2 == 0
    groups = [group_cores(cfg, g) for g in range(cfg.G)]
    C = [c for gr in groups for c in gr]
    INJ = [(cfg.inj_cols[i // 10], i % 10) for i in range(cfg.NINJ)]
    assert not set(INJ) & set(C)
    cset = crs(C)
    KSL, KTD, DC, NS = cfg.KSL, cfg.PAIRS, cfg.DCOLS, cfg.NS
    KH, LT, NW = KSL // 2, 20, 20
    DB = 3
    WGU, WDN = KSL * NW, KTD * DC
    wtile, htile, xtile, ptile = (TILE_BYTES[d] for d in (cfg.w_dt, cfg.h_dt, cfg.x_dt, cfg.p_dt))
    NT = cfg.GC // cfg.NINJ
    U, RM = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    icred = sharded_l1(device, INJ, [cfg.G * NT, 4], U, RM, torch.int32)
    warr = sharded_l1(device, C, [2, 4], U, RM, torch.int32)
    wgu_ring = sharded_l1(device, C, [2 * WGU * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    wd_ring = sharded_l1(device, C, [2 * WDN * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    hring = sharded_l1(device, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = sharded_l1(device, C, [NS * 2, 4], U, RM, torch.int32)
    xring = sharded_l1(device, C, [cfg.NXR * KSL * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags = sharded_l1(device, C, [cfg.NXR * 2, 4], U, RM, torch.int32)
    xcred = sharded_l1(device, C, [2, 4], U, RM, torch.int32)
    land = sharded_l1(device, C, [cfg.D * LT * 32, 32], cfg.p_dt, ttnn.TILE_LAYOUT)
    lflags = sharded_l1(device, C, [cfg.D * 10, 4], U, RM, torch.int32)
    lcred = sharded_l1(device, C, [10, 4], U, RM, torch.int32)
    pring = sharded_l1(device, C, [cfg.PD * NW * 32, 32], cfg.p_dt, ttnn.TILE_LAYOUT)
    scr = sharded_l1(device, C, [8, 4], U, RM, torch.int32)
    counts_l1 = sharded_l1(device, C, [1, 32 + max(4, cfg.nexp)], U, RM, torch.int32)
    idx_l1 = sharded_l1(device, C, [4, 32], U, RM, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base = a64(counts_l1.buffer_address())
    idx_addr = a64(idx_l1.buffer_address())
    pre = cfg.wsrc == "l1"
    idefs_all = [(d, "1") for d in cfg.idefs] + ([("WG_INJ_P", str(cfg.P))] if pre else [])
    idefs_all += [("WG_INJ_KS", str(KSL))]
    st = stamps_tensor(device, C)
    kdefs = ([("KS_XALT", "1")] if cfg.xalt else []) + ([("KS_FOLD2", "1")] if cfg.fold2 else [])
    kernels = []
    cp_rt = ttnn.RuntimeArgs()
    for g, gr in enumerate(groups):
        nexp_g = len(range(g, cfg.nexp, cfg.G))
        rs = noc_rects(device, gr)
        assert len(rs) == 1, rs
        r0 = list(rs[0][:4])
        r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
        cx0, ny0 = noc(device, *gr[0])
        cx1, _ = noc(device, *gr[10])
        gwr, grd = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for j, (x, y) in enumerate(gr):
            assert noc(device, x, y) == ((cx0, cx1)[j // 10], ny0 + j % 10)
            ix, iy = noc(device, *INJ[j % cfg.NINJ])
            ioff = (g * NT + j // cfg.NINJ) * 16
            cp_rt[x][y] = [st.buffer_address(), j, g, nexp_g]
            gwr[x][y] = [j, g, nexp_g, cx0, cx1, ny0, x_t.buffer_address(), idx_t.buffer_address()] + [
                counts_t.buffer_address(),
                st.buffer_address(),
            ]
            grd[x][y] = [j, g, nexp_g, cx0, cx1, ny0, ix, iy, ioff, st.buffer_address(), out_t.buffer_address()]
        common = [cfg.NXR, xring.buffer_address(), xflags.buffer_address(), xcred.buffer_address(), xtile, cfg.D]
        common += [land.buffer_address(), lflags.buffer_address(), lcred.buffer_address(), ptile, scr.buffer_address()]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("ks_writer.cpp"),
                core_ranges=crs(gr),
                compile_time_args=[KSL, KTD, cfg.GC, cfg.G, NS, hring.buffer_address(), hflags.buffer_address()]
                + r0
                + [htile, cfg.RMAX, cfg.H * 2, cnt_base, idx_addr, cfg.nexp]
                + common
                + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args()),
                runtime_args=gwr,
                defines=kdefs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("ks_reader.cpp"),
                core_ranges=crs(gr),
                compile_time_args=[KSL, KTD, DC, cfg.GC, cfg.G, NS, hflags.buffer_address()]
                + r_noc1
                + [cnt_base, icred.buffer_address(), hring.buffer_address(), htile, warr.buffer_address()]
                + [cfg.RMAX, cfg.HT]
                + common
                + [cfg.PD, pring.buffer_address()]
                + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args()),
                runtime_args=grd,
                defines=kdefs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            )
        )
    kernels += injector_kernels(device, cfg, groups, INJ, NT, wgu_t, wd_t, wgu_ring, wd_ring, icred, warr, idefs_all)
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("ks_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[KSL, KTD, DC, DB, cnt_base, cfg.G, cfg.GC, cfg.FL, cfg.TL],
            named_compile_time_args=[("expert_activation", 0)],
            defines=act_defs + [("DN_LAG", str(cfg.dn_lag))] + kdefs,
            runtime_args=cp_rt,
            config=ccfg,
        )
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, xring),
        ttnn.cb_descriptor_from_sharded_tensor(1, wgu_ring),
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        ttnn.cb_descriptor_from_sharded_tensor(3, wd_ring),
        ttnn.cb_descriptor_from_sharded_tensor(4, land),
        cb(7, ttnn.bfloat16, cfg.XRM * (KSL if cfg.xalt else KH), cset),
        cb(8, cfg.x_dt, (cfg.TL + 2) * (KSL if cfg.xalt else KH), cset),
        cb(16, cfg.h_dt, 2, cset),
        cb(17, cfg.out_dt, 2 * DC, cset),
        ttnn.cb_descriptor_from_sharded_tensor(18, pring),
    ]
    if cfg.spare_slot:  # L1 of 2 more weight slots on the group cores (the lane's 4 slots; a fit check, unused)
        cbs.append(cb(20, cfg.w_dt, 2 * (WGU + WDN), cset))
    cbs += [
        cb(0, cfg.w_dt, cfg.P * NT * (WGU + WDN) if pre else cfg.ICB * cfg.CH, crs(INJ)),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=crs(C + INJ), initial_value=0) for s in range(6)]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, idx_t, counts_t, wgu_t, wd_t, wgu_ring, wd_ring, hring, hflags, xring, xflags, xcred, land, lflags]
    io += [lcred, pring, scr, counts_l1]
    io += [idx_l1, icred, warr, st, out_t]
    return prog, io, st, {}
