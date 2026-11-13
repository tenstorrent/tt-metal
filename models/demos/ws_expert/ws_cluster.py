# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert, column-cluster scheme (scheme C, see WORKLOG.md): every core runs gate/up AND down.

  * NCL clusters, one logical column of GC (10) cores each. Core j of a cluster owns gate/up pairs
    [NPAIR j, NPAIR (j + 1)) (one 1 x 2 NPAIR fp32 DST block a row) and down columns [DCOLS j, DCOLS (j + 1)).
  * Rows go to clusters by a weighted stream-K split of the flattened (row, K-tile) work, each K-tile costing 2 NPAIR
    tile products and the row's first K-tile also its down (DCOLS x KTD): any row count balances. A row split
    between clusters k and k + 1 is finished (and its down run) by k, starting from k + 1's fp32 partial.
  * h never leaves the cluster: each core multicasts its NPAIR h tiles of a row to the column (loopback), every core
    then runs the row's down for its columns.
"""

from dataclasses import dataclass, field

import torch
import ttnn

from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1


@dataclass
class ClCfg:
    H: int = 3840
    I: int = 640
    R: int = 19
    nexp: int = 16
    NCL: int = 9  # clusters (logical columns)
    GC: int = 10  # cores a cluster (pair groups)
    # program order (order_program). "eager" + dn_lag 1 = down of row q - 1 after row q: h ring NS 4 for every M
    # (M is only known at run time). "act_hidden" hides more SiLU but its ring grows with rows per cluster.
    order: str = "eager"
    group: int = 3  # full rows between DN groups (order act_hidden)
    dn_lag: int = 1  # DN ops take rows of earlier experts or >= dn_lag productions old (0: any produced row)
    x_dt: object = ttnn.bfloat16
    w_dt: object = ttnn.bfloat4_b
    h_dt: object = ttnn.bfloat16
    out_dt: object = ttnn.bfloat16
    xt_cap: int = 150
    l1_budget: int = 1200 * 1024  # CB bytes a core (x store shrinks to fit)
    cols: tuple = None  # logical x of each cluster (default 0..NCL-1)
    glu: int = 3  # 0 stock silu + multiply; 1.. fused one pass, glu_fused.hpp MODE = glu - 1 (3: exp_21f, Prgm consts)
    cdefs: tuple = ()  # extra compute defines (debug)
    approx: bool = False  # SFPU approximate mode for the activation
    nosync: bool = False  # debug: no h exchange (the ring holds whatever is there): compute + partials only

    @property
    def KT(self):
        return self.H // 32

    @property
    def PAIRS(self):
        return self.I // 32

    @property
    def NPAIR(self):
        return self.PAIRS // self.GC

    @property
    def GW(self):
        return 2 * self.NPAIR

    @property
    def HT(self):
        return self.H // 32

    @property
    def DCOLS(self):
        return self.HT // self.GC

    def col(self, k):
        return self.cols[k] if self.cols else k


@dataclass
class ClPlan:
    cfg: ClCfg
    starts: list
    pieces: list  # per cluster [(row, k0, k1, flags, xoff)]
    owned: list  # per cluster: rows it finishes, in order
    XT: int = 0
    cost: list = field(default_factory=list)
    NS: int = 4


PRE_L, PRE_R, SEND_L, SEND_R, DN = 1, 2, 4, 8, 16


def order_program(ps, order="act_hidden", group=3):
    """A cluster's per-expert program: its gate/up pieces plus one DN op per owned row.

    act_hidden: [full rows][send pieces][DN ops][pieces that wait for a partial]. Every activation is followed by a
    long matmul block (the next full row, a send piece, or the next expert's first piece) instead of short down
    blocks, which would wait for the DST half the activation still holds; senders still send before every receiver
    (receivers come last), so no chain of waits forms across clusters.
    eager: [send pieces][full rows][receivers], a DN op right after each produced row (down of the previous row)."""
    S = [pc for pc in ps if pc[3] & (SEND_L | SEND_R)]
    F = [pc for pc in ps if not pc[3] & (SEND_L | SEND_R | PRE_L | PRE_R)]
    P = [pc for pc in ps if pc[3] & (PRE_L | PRE_R) and not pc[3] & (SEND_L | SEND_R)]
    mid = [pc for pc in S if pc[3] & (PRE_L | PRE_R)]  # chain middles (small R): send after receiving
    S = [pc for pc in S if pc not in mid]
    dn = (-1, 0, 0, DN, 0)
    n_own = len(F) + len(P)
    if order == "eager":
        prog = list(S)
        for pc in F + P:
            prog += [pc, dn]
        return prog + mid
    # large R: DN ops after every `group` full rows (one exposed activation per group, bounded ring)
    prog = []
    for i in range(0, len(F), group):
        chunk = F[i : i + group]
        prog += chunk
        if i + group < len(F):
            prog += [dn] * len(chunk)
    n_left = n_own - sum(1 for op in prog if op[3] & DN)
    return prog + S + [dn] * n_left + P + mid


def ring_depth(prog, nexp=4, lag=0):
    """h ring rows (NS) the no-credit scheme needs: a peer writes h(q) into slot q % NS once it has consumed c rows,
    which needed this core's h(c - 1), produced after this core consumed c' rows: slot q % NS must not still hold an
    unconsumed row, so NS >= q + 1 - c'. Every core of a cluster runs the same program."""
    produced = consumed = 0
    cons_at = {}  # q -> rows consumed when h(q) is produced
    for _ in range(nexp):
        e_start = produced
        for op in prog:
            if op[3] & DN:
                if consumed < produced and (not lag or consumed < e_start or consumed + lag < produced):
                    consumed += 1
            elif not op[3] & (SEND_L | SEND_R):
                cons_at[produced] = consumed
                produced += 1
    need = 1
    for q, c in cons_at.items():
        if c == 0:
            need = max(need, q + 1)
            continue
        need = max(need, q + 1 - cons_at[c - 1])
    return need


def plan(cfg: ClCfg) -> ClPlan:
    """Split the flattened (row, K-tile) units into NCL contiguous ranges and pick each split row's owner.

    Cost per core: g = 2 NPAIR tile products a K-tile, d = DCOLS x KTD a row's down (paid by the row's owner). A row
    split between clusters c and c + 1 may be owned by either (its partial flows towards the owner): the boundary and
    the owner are chosen together so the cumulative cost tracks the target. Rows spanning 3+ clusters (small R) are
    owned by their leftmost cluster (partials flow left along the chain).
    Execution order in a cluster: pieces that only send first, then whole rows, then pieces that wait for a partial
    (senders send early, receivers use them last: no stalls)."""
    KT, R, N = cfg.KT, cfg.R, cfg.NCL
    g, d = cfg.GW, cfg.DCOLS * cfg.PAIRS
    U = R * KT
    total = R * (KT * g + d)
    owner = {}  # row -> cluster
    starts = [0]
    s, cum = 0, 0.0
    for c in range(N - 1):
        target = (c + 1) * total / N
        row_s = s // KT
        inherited = 1 if (s % KT and owner.get(row_s) == c) else 0
        best = None
        for e in range(s + 1, U + 1):
            started = [r for r in range(-(-s // KT), -(-e // KT)) if s <= r * KT < e]
            split_row = e // KT if (e % KT and e < U) else None
            opts = [(0, c)]
            # give the split row (and its down) to the next cluster, if its rest fits well inside that cluster's
            # range (else the row would span 3 clusters and its owner sit in the middle)
            if split_row is not None and split_row in started and (split_row + 1) * KT - e < 0.5 * total / N / g:
                opts.append((1, c + 1))
            for give, own_split in opts:
                cost = cum + g * (e - s) + d * (inherited + len(started) - give)
                err = abs(cost - target)
                if best is None or err < best[0]:
                    best = (err, e, own_split, cost, started, split_row)
            if cum + g * (e - s) > target + d + g:
                break
        _, e, own_split, cost, started, split_row = best
        for r in started:
            owner.setdefault(r, c)
        if split_row is not None and split_row in started:
            owner[split_row] = own_split
        cum = cost
        starts.append(e)
        s = e
        if s >= U:
            break
    while len(starts) < N + 1:
        starts.append(U)
    starts[-1] = U
    for r in range(R):
        owner.setdefault(r, N - 1)
    pieces, owned, cost = [], [], []
    for i in range(N):
        ps = []
        f = starts[i]
        while f < starts[i + 1]:
            r = f // KT
            end = min(starts[i + 1], (r + 1) * KT)
            k0, k1 = f - r * KT, end - r * KT
            o = owner[r]
            fl = 0
            if o < i:  # partial flows left (chain: first take the right part's partial)
                fl |= SEND_L | (PRE_R if k1 < KT else 0)
            elif o > i:
                fl |= SEND_R | (PRE_L if k0 > 0 else 0)
            else:
                fl |= (PRE_L if k0 > 0 else 0) | (PRE_R if k1 < KT else 0)
            assert (fl & 3) != 3, "a piece can start from one partial only"
            ps.append((r, k0, k1, fl, f - starts[i]))
            f = end
        ps.sort(key=lambda pc: pc[4])
        prog = order_program(ps, cfg.order, cfg.group)
        pieces.append(prog)
        owned.append([pc[0] for pc in prog if not (pc[3] & (SEND_L | SEND_R | DN))])
        cost.append(g * (starts[i + 1] - starts[i]) + d * len(owned[-1]))
    NS = max(2, max(ring_depth(pr, lag=cfg.dn_lag) for pr in pieces))
    # L1: the x store gets what the rest leaves (bench only: in the op x streams through a small ring)
    from models.demos.ws_expert.bench_util import TILE_BYTES as TB

    hb, wb = TB[cfg.h_dt], TB[cfg.w_dt]
    rest = (
        cfg.KT * cfg.GW * wb
        + cfg.PAIRS * cfg.DCOLS * wb
        + NS * cfg.PAIRS * hb
        + 2 * 2 * cfg.GW * 4096
        + cfg.GW * 4096
        + 2 * cfg.NPAIR * hb
        + 2 * cfg.DCOLS * TB[cfg.out_dt]
    )
    xt_fit = max(16, (cfg.l1_budget - rest) // TB[cfg.x_dt])
    XT = min(max(1, max(starts[i + 1] - starts[i] for i in range(N))), cfg.xt_cap, xt_fit)
    return ClPlan(cfg, starts, pieces, owned, XT, cost, NS)


def effective_x(p: ClPlan, stores):
    cfg = p.cfg
    X = torch.zeros(cfg.R * 32, cfg.KT * 32)
    for i in range(cfg.NCL):
        for f in range(p.starts[i], p.starts[i + 1]):
            r, k = divmod(f, cfg.KT)
            X[r * 32 : r * 32 + 32, k * 32 : k * 32 + 32] = stores[i, (f - p.starts[i]) % p.XT]
    return X


def cores(cfg):
    return [(cfg.col(k), j) for k in range(cfg.NCL) for j in range(cfg.GC)]


def build(device, cfg: ClCfg, p: ClPlan, x_t, w_t, wd_t, out_t):
    C = cores(cfg)
    cset = crs(C)
    KT, KTD, GW, NPAIR, DC = cfg.KT, cfg.PAIRS, cfg.GW, cfg.NPAIR, cfg.DCOLS
    htile = TILE_BYTES[cfg.h_dt]
    hring = sharded_l1(device, C, [p.NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    flags = sharded_l1(device, C, [p.NS * cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    pinl = sharded_l1(device, C, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    pinr = sharded_l1(device, C, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    st = stamps_tensor(device, C)

    cp_rt, wr_rt, rd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    col_rects = {}
    for k in range(cfg.NCL):
        x = cfg.col(k)
        rs = noc_rects(device, [(x, j) for j in range(cfg.GC)])
        assert len(rs) == 1
        col_rects[k] = rs[0][:4]
        ps, own = p.pieces[k], p.owned[k]
        for j in range(cfg.GC):
            lft = noc(device, cfg.col(k - 1), j) if k > 0 else (0, 0)
            rgt = noc(device, cfg.col(k + 1), j) if k + 1 < cfg.NCL else (0, 0)
            cp = [st.buffer_address(), len(ps)]
            for r, k0, k1, fl, xoff in ps:  # (DN ops: zeros + the DN flag)
                cp += [k0, k1, fl, xoff]
            cp_rt[x][j] = cp
            wr_rt[x][j] = (
                [len(ps), j, lft[0], lft[1], out_t.buffer_address(), max(1, len(own)), rgt[0], rgt[1]]
                + (own if own else [0])
                + [pc[3] for pc in ps]
            )
            rd_rt[x][j] = [
                x_t.buffer_address(),
                w_t.buffer_address(),
                wd_t.buffer_address(),
                k,
                int(k > 0),
                rgt[0],
                rgt[1],
                j,
                len(ps),
                int(k + 1 < cfg.NCL),
                lft[0],
                lft[1],
            ] + [pc[3] for pc in ps]
    # the writer's column rectangle is per cluster: compile-time, so one kernel per cluster
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
    utd[6] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=cfg.approx
    )
    ccfg.unpack_to_dest_mode = utd
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kpath("wc_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, NPAIR, p.XT, cfg.nexp, KTD, DC],
            named_compile_time_args=[("expert_activation", 0)],
            defines=act_defs + [(d, "1") for d in cfg.cdefs] + [("WC_GLU", str(cfg.glu)), ("DN_LAG", str(int(cfg.dn_lag)))],
            runtime_args=cp_rt,
            config=ccfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wc_reader.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, GW, p.XT, cfg.PAIRS * 2, cfg.nexp, KTD, DC, cfg.HT, p.NS, cfg.GC]
            + [flags.buffer_address()]
            + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(w_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args()),
            runtime_args=rd_rt,
            defines=([("WC_NOSYNC", "1")] if cfg.nosync else []) + [("DN_LAG", str(int(cfg.dn_lag)))],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
    ]
    for k in range(cfg.NCL):
        x = cfg.col(k)
        kr = ttnn.RuntimeArgs()
        for j in range(cfg.GC):
            kr[x][j] = wr_rt[x][j]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("wc_writer.cpp"),
                core_ranges=crs([(x, j) for j in range(cfg.GC)]),
                compile_time_args=[NPAIR, p.NS, KTD, cfg.GC, cfg.nexp]
                + [hring.buffer_address(), flags.buffer_address(), pinl.buffer_address()]
                + list(col_rects[k])
                + [htile, DC, cfg.HT, pinr.buffer_address()]
                + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args()),
                runtime_args=kr,
                defines=([("WC_NOSYNC", "1")] if cfg.nosync else []) + [("DN_LAG", str(int(cfg.dn_lag)))],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        )
    cbs = [
        cb(0, cfg.x_dt, p.XT, cset),
        cb(1, cfg.w_dt, KT * GW, cset),
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        cb(3, cfg.w_dt, KTD * DC, cset),
        cb(4, ttnn.float32, GW, cset),
        ttnn.cb_descriptor_from_sharded_tensor(5, pinl),
        ttnn.cb_descriptor_from_sharded_tensor(6, pinr),
        cb(16, cfg.h_dt, 2 * NPAIR, cset),
        cb(17, cfg.out_dt, 2 * DC, cset),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(5)]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, w_t, wd_t, hring, flags, pinl, pinr, st, out_t]
    return prog, io, st
