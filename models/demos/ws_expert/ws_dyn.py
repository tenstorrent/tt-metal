# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert with dynamic M: every expert's schedule is computed on device from its row count (kernels/ws_plan.hpp;
plan_py below mirrors it for sizing and checks). Inputs per call: counts [NEXP] (uint32, one DRAM page) and row index
tables [NEXP, RMAX * 32] (one page per expert, padding rows -> the zero row T of x). x streamed and tilized as in
ws_x.py; weights resident. Ring sizes do not depend on M: h ring 4 rows, x ring NX blocks, x staging sized for MMAX.
"""

from dataclasses import dataclass

import torch
import ttnn

from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1

PRE_L, PRE_R, SEND_L, SEND_R, DN = 1, 2, 4, 8, 16
NSLOT, SLOT_WORDS = 3, 512
PLAN_WORDS = NSLOT * SLOT_WORDS + 16


@dataclass
class DCfg:
    H: int = 3840
    I: int = 640
    nexp: int = 16
    NCL: int = 9
    GC: int = 10
    BK: int = 12
    NX: int = 15
    MMAX: int = 5120
    T: int = 8192
    LOOK: int = 30  # compute tilizes its own x blocks up to LOOK blocks past its consumption point
    glu: int = 3
    dn_lag: int = 1
    x_dt: object = ttnn.bfloat16
    w_dt: object = ttnn.bfloat4_b
    h_dt: object = ttnn.bfloat16
    out_dt: object = ttnn.bfloat16

    KT = property(lambda s: s.H // 32)
    PAIRS = property(lambda s: s.I // 32)
    NPAIR = property(lambda s: s.PAIRS // s.GC)
    GW = property(lambda s: 2 * s.NPAIR)
    HT = property(lambda s: s.H // 32)
    DCOLS = property(lambda s: s.HT // s.GC)
    RMAX = property(lambda s: (s.MMAX + 31) // 32)


def boundary(cfg, R, na, c):
    KT, G, D = cfg.KT, cfg.GW, cfg.DCOLS * cfg.PAIRS
    rowcost = KT * G + D
    t = (c * R * rowcost) // na
    r, q = divmod(t, rowcost)
    if q >= D + G and (q - D) // G <= KT - 1:
        return r * KT + (q - D) // G, 0
    if 1 <= q // G <= KT - 1:
        return r * KT + q // G, 1
    if q < G:
        return r * KT, 0
    return (r + 1) * KT, 0


def plan_py(cfg, M, k, j):
    """Mirror of ws_plan::make: (ops [(k0, k1, flags, row)], own rows, nbe, my blocks [(l, r, k, n)])."""
    R = (M + 31) // 32
    na = min(cfg.NCL, max(1, R // 2)) if R else 0
    ops, own, my, nbe = [], [], [], 0
    if k >= na:
        return ops, own, 0, my
    s, sr = boundary(cfg, R, na, k) if k > 0 else (0, 0)
    e, er = boundary(cfg, R, na, k + 1) if k + 1 < na else (R * cfg.KT, 0)
    pcs = []
    f = s
    while f < e:
        r = f // cfg.KT
        end = min((r + 1) * cfg.KT, e)
        k0, k1 = f - r * cfg.KT, end - r * cfg.KT
        fl = 0
        if k0 > 0:
            fl |= PRE_L if sr else SEND_L
        if k1 < cfg.KT:
            fl |= SEND_R if er else PRE_R
        pcs.append((k0, k1, fl, r))
        f = end
    for ps in range(3):
        for k0, k1, fl, r in pcs:
            cls = 0 if fl & (SEND_L | SEND_R) else (2 if fl & (PRE_L | PRE_R) else 1)
            if cls != ps:
                continue
            ops.append((k0, k1, fl, r))
            for kk in range(k0, k1, cfg.BK):
                if nbe % cfg.GC == j:
                    my.append((nbe, r, kk, min(cfg.BK, k1 - kk)))
                nbe += 1
            if ps > 0:
                own.append(r)
                ops.append((0, 0, DN, -1))
    return ops, own, nbe, my


def check_plans(cfg, M):
    """Every K-tile of every row covered once, every row owned once, partial directions consistent."""
    R = (M + 31) // 32
    cover = torch.zeros(R, cfg.KT, dtype=torch.int32)
    owners = torch.zeros(R, dtype=torch.int32)
    for k in range(cfg.NCL):
        ops, own, _, _ = plan_py(cfg, M, k, 0)
        for k0, k1, fl, r in ops:
            if fl & DN:
                continue
            cover[r, k0:k1] += 1
        for r in own:
            owners[r] += 1
    assert bool((cover == 1).all()), "K coverage"
    assert bool((owners == 1).all()), "row ownership"


def xst_blocks(cfg):
    """x staging: own blocks tilized ahead (window LOOK) plus those waiting for a ring slot, any M."""
    return -(-cfg.LOOK // cfg.GC) + 4


def xst_blocks_expert_batch(cfg):
    """(old: one batch an expert) x staging blocks so the next expert's tilize batch always fits (any M <= MMAX)."""
    need = 1
    for M in range(1, cfg.MMAX + 1):
        for k in range(cfg.NCL):
            nbe = plan_py(cfg, M, k, 0)[2]
            if nbe == 0:
                continue
            own_max = -(-nbe // cfg.GC)
            need = max(need, own_max + -(-max(0, nbe - cfg.NX) // cfg.GC) + 1)
    return need


def build(device, cfg: DCfg, x_t, idx_t, counts_t, w_t, wd_t, out_t, XST):
    C = [(k, j) for k in range(cfg.NCL) for j in range(cfg.GC)]
    cset = crs(C)
    KT, KTD, GW, NPAIR, DC, BK, NX, NS = cfg.KT, cfg.PAIRS, cfg.GW, cfg.NPAIR, cfg.DCOLS, cfg.BK, cfg.NX, 4
    htile, xtile = TILE_BYTES[cfg.h_dt], TILE_BYTES[cfg.x_dt]
    hring = sharded_l1(device, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = sharded_l1(device, C, [NS * cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xring = sharded_l1(device, C, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags = sharded_l1(device, C, [NX, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xprog = sharded_l1(device, C, [cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    pinl = sharded_l1(device, C, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    pinr = sharded_l1(device, C, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    idx_l1 = sharded_l1(device, C, [2, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)  # one row tile's ids
    planbuf = sharded_l1(device, C, [1, PLAN_WORDS], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    counts_l1 = sharded_l1(device, C, [1, max(4, cfg.nexp)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    st = stamps_tensor(device, C)
    lag = [("DN_LAG", str(int(cfg.dn_lag)))]
    kernels = []
    cp_rt, rd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k in range(cfg.NCL):
        col = [(k, j) for j in range(cfg.GC)]
        rs = noc_rects(device, col)
        assert len(rs) == 1
        r0 = list(rs[0][:4])
        r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
        wr_rt = ttnn.RuntimeArgs()
        for j in range(cfg.GC):
            lft = noc(device, k - 1, j) if k > 0 else (0, 0)
            rgt = noc(device, k + 1, j) if k + 1 < cfg.NCL else (0, 0)
            cp_rt[k][j] = [st.buffer_address()]
            wr_rt[k][j] = [j, lft[0], lft[1], rgt[0], rgt[1], out_t.buffer_address(), x_t.buffer_address()] + [
                idx_t.buffer_address(),
                counts_t.buffer_address(),
                k,
            ]
            c0 = noc(device, k, 0)
            assert all(noc(device, k, jj) == (c0[0], c0[1] + jj) for jj in range(cfg.GC))
            rd_rt[k][j] = [w_t.buffer_address(), wd_t.buffer_address(), int(k > 0), rgt[0], rgt[1], j] + [
                int(k + 1 < cfg.NCL),
                lft[0],
                lft[1],
                c0[0],
                c0[1],
            ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wd_writer.cpp"),
                core_ranges=crs(col),
                compile_time_args=[NPAIR, NS, KTD, cfg.GC, cfg.nexp]
                + [hring.buffer_address(), hflags.buffer_address(), pinl.buffer_address()]
                + r0
                + [htile, DC, cfg.HT, pinr.buffer_address(), BK, cfg.RMAX, cfg.H * 2, idx_l1.buffer_address()]
                + [planbuf.buffer_address(), counts_l1.buffer_address(), KT, cfg.NCL]
                + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args()),
                runtime_args=wr_rt,
                defines=lag,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        )
        rk = ttnn.RuntimeArgs()
        for j in range(cfg.GC):
            rk[k][j] = rd_rt[k][j]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wd_reader.cpp"),
                core_ranges=crs(col),
                compile_time_args=[KT, GW, cfg.PAIRS * 2, cfg.nexp, KTD, DC, cfg.HT, NS, cfg.GC]
                + [hflags.buffer_address(), BK, NX, xring.buffer_address(), xflags.buffer_address()]
                + r_noc1
                + [xtile, xprog.buffer_address(), planbuf.buffer_address()]
                + list(ttnn.TensorAccessorArgs(w_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args()),
                runtime_args=rk,
                defines=lag,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            )
        )
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
    utd[6] = ttnn.UnpackToDestMode.UnpackToDestFp32
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
            kernel_source=kpath("wd_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, NPAIR, BK, cfg.nexp, KTD, DC, planbuf.buffer_address(), NX, cfg.LOOK],
            named_compile_time_args=[("expert_activation", 0)],
            defines=act_defs
            + lag
            + [("WC_GLU", str(cfg.glu))]
            + ([("WX_X_FMT_DIFFERS", "1")] if cfg.x_dt != cfg.h_dt else []),
            runtime_args=cp_rt,
            config=ccfg,
        )
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, xring),
        cb(1, cfg.w_dt, KT * GW, cset),
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        cb(3, cfg.w_dt, KTD * DC, cset),
        cb(4, ttnn.float32, GW, cset),
        ttnn.cb_descriptor_from_sharded_tensor(5, pinl),
        ttnn.cb_descriptor_from_sharded_tensor(6, pinr),
        cb(7, ttnn.bfloat16, 3 * BK, cset),
        cb(8, cfg.x_dt, XST * BK, cset),
        cb(16, cfg.h_dt, 2 * NPAIR, cset),
        cb(17, cfg.out_dt, 2 * DC, cset),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(8)]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, idx_t, counts_t, w_t, wd_t, hring, hflags, xring, xflags, xprog, pinl, pinr, idx_l1, planbuf]
    io += [counts_l1, st, out_t]
    return prog, io, st
