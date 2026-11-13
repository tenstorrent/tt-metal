# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert column clusters with x streamed from DRAM (row-major bf16, gathered by a row index table) and tilized on
the cluster's own cores (kernels wx_*.cpp). Weights stay resident (stage 2 of the op: x; weights come next).

Per cluster, the x stream (its gate/up pieces' K ranges, in program order) is cut into blocks of BK K-tiles; block j
of an expert is gathered (BRISC), tilized (compute, one gate/up piece ahead of its use) and multicast to the column
(NCRISC, NOC1) by core j % GC. The x ring has NX (a multiple of GC) block slots on every core; consumers credit the
producer of the block that reuses a slot.
"""

from dataclasses import dataclass

import torch
import ttnn

from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_cluster import DN, SEND_L, SEND_R, ClCfg, cores, plan
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1

TIL, TIL_NEXT = 32, 64


@dataclass
class XCfg(ClCfg):
    BK: int = 12  # x block: K-tiles
    NX: int = 20  # x ring blocks
    T: int = 4096  # tokens in the x tensor (plus one zero row)
    wdefs: tuple = ()  # debug defines for the BRISC (WX_NO_GATHER)
    til: str = "expert"  # "expert": one tilize batch an expert (the next expert's own blocks); "piece": one piece ahead
    XST: int = 0  # tilized-block staging (blocks); 0: sized by the host (deadlock-free, see build)
    XRM: int = 3  # row-major (gathered) staging (blocks)


def x_schedule(cfg: XCfg, prog):
    if cfg.til == "expert":
        return x_schedule_expert(cfg, prog)
    return x_schedule_piece(cfg, prog)


def x_schedule_expert(cfg: XCfg, prog):
    """One TILIZE batch at the start of each expert: this core's blocks of the next expert (prologue: expert 0's)."""
    gus = [op for op in prog if not op[3] & DN]
    blocks = []
    for g, (r, k0, k1, fl, _) in enumerate(gus):
        for k in range(k0, k1, cfg.BK):
            blocks.append((r, k, min(cfg.BK, k1 - k), g))
    own = {j: [b for b in range(len(blocks)) if b % cfg.GC == j] for j in range(cfg.GC)}
    progs = {j: [(-1, 0, 0, TIL_NEXT, 0)] * len(own[j]) + list(prog) for j in range(cfg.GC)}
    pro = {j: len(own[j]) for j in range(cfg.GC)}
    return blocks, own, progs, pro


def x_schedule_piece(cfg: XCfg, prog):
    """A cluster's x blocks (stream order) and, per core, its compute program with TILIZE ops one piece ahead."""
    gus = [op for op in prog if not op[3] & DN]
    blocks = []  # (r, k, n, piece)
    for g, (r, k0, k1, fl, _) in enumerate(gus):
        for k in range(k0, k1, cfg.BK):
            blocks.append((r, k, min(cfg.BK, k1 - k), g))
    own = {j: [b for b in range(len(blocks)) if b % cfg.GC == j] for j in range(cfg.GC)}
    progs, pro = {}, {}
    for j in range(cfg.GC):
        mine = own[j]
        ops, g = [], 0
        for op in prog:
            if not op[3] & DN:
                nxt = g + 1
                if nxt < len(gus):
                    ops += [(-1, 0, 0, TIL, 0)] * sum(1 for b in mine if blocks[b][3] == nxt)
                else:
                    ops += [(-1, 0, 0, TIL_NEXT, 0)] * sum(1 for b in mine if blocks[b][3] == 0)
                g += 1
            ops.append(op)
        progs[j] = ops
        pro[j] = sum(1 for b in mine if blocks[b][3] == 0)
    return blocks, own, progs, pro


def build(device, cfg: XCfg, p, x_t, idx_t, w_t, wd_t, out_t):
    C = cores(cfg)
    cset = crs(C)
    KT, KTD, GW, NPAIR, DC, BK, NX = cfg.KT, cfg.PAIRS, cfg.GW, cfg.NPAIR, cfg.DCOLS, cfg.BK, cfg.NX
    htile, xtile = TILE_BYTES[cfg.h_dt], TILE_BYTES[cfg.x_dt]
    hring = sharded_l1(device, C, [p.NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = sharded_l1(device, C, [p.NS * cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xring = sharded_l1(device, C, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags = sharded_l1(device, C, [NX, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    xprog = sharded_l1(device, C, [cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    pinl = sharded_l1(device, C, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    pinr = sharded_l1(device, C, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    idx_l1 = sharded_l1(device, C, [1, cfg.R * 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    st = stamps_tensor(device, C)
    dn_lag = [("DN_LAG", str(int(cfg.dn_lag)))]

    cp_rt, rd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    kernels = []
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
    utd[6] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    ccfg.unpack_to_dest_mode = utd
    sched = {}
    # staging: the whole next-expert batch must fit beside this expert's blocks still waiting for a ring slot, else
    # compute blocks inside the tilize batch and never frees the slots they wait for
    xst_need = 1
    for k in range(cfg.NCL):
        nbe = len(x_schedule(cfg, p.pieces[k])[0])
        own_max = -(-nbe // cfg.GC)
        xst_need = max(xst_need, own_max + -(-max(0, nbe - NX) // cfg.GC) + 1)
    XST = cfg.XST or xst_need
    assert XST >= xst_need, (XST, xst_need)
    for k in range(cfg.NCL):
        x = cfg.col(k)
        col = [(x, j) for j in range(cfg.GC)]
        rs = noc_rects(device, col)
        assert len(rs) == 1
        r0 = list(rs[0][:4])
        r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
        col_noc = [v for c in col for v in noc(device, *c)]
        prog, own_rows = p.pieces[k], p.owned[k]
        blocks, own, progs, pro = x_schedule(cfg, prog)
        sched[k] = (blocks, own, progs, pro)
        wr_rt = ttnn.RuntimeArgs()
        for j in range(cfg.GC):
            lft = noc(device, cfg.col(k - 1), j) if k > 0 else (0, 0)
            rgt = noc(device, cfg.col(k + 1), j) if k + 1 < cfg.NCL else (0, 0)
            ops = progs[j]
            cp = [st.buffer_address(), len(ops)]
            for r, k0, k1, fl, _ in ops:
                cp += [k0, k1, fl, 0]
            cp += [pro[j]]
            cp_rt[x][j] = cp
            mine = own[j]
            wr_rt[x][j] = (
                [len(ops), j, lft[0], lft[1], out_t.buffer_address(), max(1, len(own_rows)), rgt[0], rgt[1]]
                + [x_t.buffer_address(), idx_t.buffer_address(), len(mine)]
                + (own_rows if own_rows else [0])
                + [op[3] for op in ops]
                + [v for b in mine for v in blocks[b][:3]]
            )
            rd_rt[x][j] = (
                [w_t.buffer_address(), wd_t.buffer_address(), int(k > 0), rgt[0], rgt[1], j, len(ops)]
                + [int(k + 1 < cfg.NCL), lft[0], lft[1], len(mine), len(blocks)]
                + col_noc
                + [op[3] for op in ops]
                + mine
            )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wx_writer.cpp"),
                core_ranges=crs(col),
                compile_time_args=[NPAIR, p.NS, KTD, cfg.GC, cfg.nexp]
                + [hring.buffer_address(), hflags.buffer_address(), pinl.buffer_address()]
                + r0
                + [htile, DC, cfg.HT, pinr.buffer_address(), BK, cfg.R, cfg.H * 2, idx_l1.buffer_address()]
                + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args()),
                runtime_args=wr_rt,
                defines=dn_lag + [(d, "1") for d in cfg.wdefs],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wx_reader.cpp"),
                core_ranges=crs(col),
                compile_time_args=[KT, GW, cfg.PAIRS * 2, cfg.nexp, KTD, DC, cfg.HT, p.NS, cfg.GC]
                + [hflags.buffer_address(), BK, NX, xring.buffer_address(), xflags.buffer_address()]
                + r_noc1
                + [xtile, xprog.buffer_address()]
                + list(ttnn.TensorAccessorArgs(w_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args()),
                runtime_args=_subset(rd_rt, col),
                defines=dn_lag,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            )
        )
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=kpath("wx_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, NPAIR, BK, cfg.nexp, KTD, DC],
            named_compile_time_args=[("expert_activation", 0)],
            defines=act_defs
            + dn_lag
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
        cb(7, ttnn.bfloat16, cfg.XRM * BK, cset),  # (row-major bf16 rows as gathered)
        cb(8, cfg.x_dt, XST * BK, cset),
        cb(16, cfg.h_dt, 2 * NPAIR, cset),
        cb(17, cfg.out_dt, 2 * DC, cset),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(7)]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, idx_t, w_t, wd_t, hring, hflags, xring, xflags, xprog, pinl, pinr, idx_l1, st, out_t]
    build.last_xst = XST
    return prog, io, st, sched


def _subset(rt, cores_):
    out = ttnn.RuntimeArgs()
    for x, y in cores_:
        out[x][y] = rt[x][y]
    return out
