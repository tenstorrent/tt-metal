# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Weight-streaming routed expert (ws_expert): one expert's gate/up -> SwiGLU -> down on one Blackhole chip, as a
generic_op program (stage 1: x and weights resident in L1, h exchanged on chip).

Layout (scheme B, see WORKLOG.md):
  * gate/up cores: a GR x GC grid (logical x = grid row i < GR, y = pair group j < GC). Core (i, j) owns NPAIR = PAIRS
    / GC gate/up pairs (1 x 2 NPAIR fp32 DST block) and the i-th of GR equal shares of the expert's flattened
    (row, K-tile) work (stream-K): any row count balances; a row split across two cores is finished by the earlier
    one, which starts from the later one's fp32 partial (exact: unpacked to fp32 DST).
  * down cores: ND cores right of the grid, each CW = HT / ND output columns of every row (1 x CW fp32 DST block).
  * h: every h row is multicast by its gate/up producers into an NS-row ring on all down cores; per-slot flag words
    (q + 1) mark arrival; a down leader aggregates slot releases and publishes CONS to the gate/up cores.
"""

from dataclasses import dataclass, field

import torch
import ttnn

from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor

SEM_CONS, SEM_GRANT, SEM_ARR, SEM_SCR, SEM_CREDIT = range(5)


@dataclass
class WsCfg:
    H: int = 3840
    I: int = 640
    R: int = 19  # row tiles (M = 32 R)
    nexp: int = 16  # experts run back to back (same shape; resident data reused)
    GR: int = 8  # gate/up grid rows (stream-K split of each pair group's rows)
    GC: int = 10  # gate/up pair groups
    ND: int = 40  # down cores
    NS: int = 8  # h ring rows on the down cores
    x_dt: object = ttnn.bfloat16
    w_dt: object = ttnn.bfloat4_b
    h_dt: object = ttnn.bfloat16
    out_dt: object = ttnn.bfloat16
    xt_cap: int = 400  # x store tiles a gate/up core (L1 budget; past it the store wraps)
    grid_x0: int = 0
    debug_no_send: bool = False  # gate/up cores drop h (down cores idle): gate/up compute alone

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
    def CW(self):
        return self.HT // self.ND


@dataclass
class Plan:
    cfg: WsCfg
    starts: list  # flattened work start per grid row (len GR + 1)
    pieces: list  # per grid row: [(row, k0, k1, flags, xoff)]
    order: list  # down order: place -> row
    qpos: dict = field(default_factory=dict)  # row -> place
    XT: int = 0


def plan(cfg: WsCfg) -> Plan:
    KT, R, GR = cfg.KT, cfg.R, cfg.GR
    total = R * KT
    starts = [(i * total) // GR for i in range(GR + 1)]
    pieces = []
    finish = {}
    for i in range(GR):
        ps = []
        f = starts[i]
        while f < starts[i + 1]:
            r = f // KT
            end = min(starts[i + 1], (r + 1) * KT)
            k0, k1 = f - r * KT, end - r * KT
            fl = (1 if k1 < KT else 0) | (2 if k0 > 0 else 0)
            ps.append((r, k0, k1, fl, f - starts[i]))
            if k0 == 0:  # the row's owner finishes it at the end of this piece
                finish[r] = end - starts[i]
            f = end
        pieces.append(ps)
    order = sorted(range(R), key=lambda r: (finish[r], r))
    qpos = {r: p for p, r in enumerate(order)}
    XT = min(max(starts[i + 1] - starts[i] for i in range(GR)), cfg.xt_cap)
    return Plan(cfg, starts, pieces, order, qpos, XT)


def effective_x_tiles(p: Plan, stores):
    """x as the device computes it: tile (r, k) is store_i[(f - s_i) % XT] of its grid row i. stores: [GR, XT, 32, 32]."""
    cfg = p.cfg
    X = torch.zeros(cfg.R * 32, cfg.KT * 32)
    for i in range(cfg.GR):
        for f in range(p.starts[i], p.starts[i + 1]):
            r, k = divmod(f, cfg.KT)
            X[r * 32 : r * 32 + 32, k * 32 : k * 32 + 32] = stores[i, (f - p.starts[i]) % p.XT]
    return X


def gu_cores(cfg):
    return [(cfg.grid_x0 + i, j) for i in range(cfg.GR) for j in range(cfg.GC)]


def dn_cores(cfg):
    x0 = cfg.grid_x0 + cfg.GR
    return [(x0 + c // 10, c % 10) for c in range(cfg.ND)]


def noc(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return c.x, c.y


def noc_rects(device, cores):
    """Split a logical rectangle of cores into rectangles contiguous in NoC coordinates (Blackhole's Tensix columns
    jump over the DRAM / L2CPU columns): [(x0, y0, x1, y1, n_cores)]."""
    xs = sorted({c[0] for c in cores})
    ys = sorted({c[1] for c in cores})
    assert len(cores) == len(xs) * len(ys), "multicast targets must be a rectangle"
    runs, cur = [], [xs[0]]
    for x in xs[1:]:
        if noc(device, x, ys[0])[0] == noc(device, cur[-1], ys[0])[0] + 1:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    out = []
    for run in runs:
        a = noc(device, run[0], ys[0])
        b = noc(device, run[-1], ys[-1])
        out.append((a[0], a[1], b[0], b[1], len(run) * len(ys)))
    return out


def rect_noc(device, cores):
    rs = noc_rects(device, cores)
    assert len(rs) == 1, f"multicast rectangle crosses a NoC column gap: {rs}"
    return list(rs[0][:4])


def _unused_rect_noc(device, cores):
    xs = [c[0] for c in cores]
    ys = [c[1] for c in cores]
    a = noc(device, min(xs), min(ys))
    b = noc(device, max(xs), max(ys))
    assert len(cores) == (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1), "multicast targets must be a rectangle"
    return [a[0], a[1], b[0], b[1]]


def sharded_l1(device, cores, shard_shape, dtype, layout, torch_dtype=torch.float32):
    grid = crs(cores)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = torch.zeros([len(cores) * shard_shape[0], shard_shape[1]], dtype=torch_dtype)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mem)


def build(device, cfg: WsCfg, p: Plan, x_t, w_t, wd_t, out_t):
    """generic_op program + its io tensors (the last is the output)."""
    G, D = gu_cores(cfg), dn_cores(cfg)
    allc = G + D
    gset, dset, aset = crs(G), crs(D), crs(allc)
    KT, KTD, GW, NPAIR, CW = cfg.KT, cfg.PAIRS, cfg.GW, cfg.NPAIR, cfg.CW
    htile = TILE_BYTES[cfg.h_dt]
    # L1 tensors written remotely (fixed addresses on every core)
    hring = sharded_l1(device, D, [cfg.NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    flags = sharded_l1(device, D, [cfg.NS * cfg.GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    pin = sharded_l1(device, G, [2 * GW * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    st = stamps_tensor(device, allc)
    drect = rect_noc(device, D)
    grects = noc_rects(device, G)
    assert len(grects) <= 2
    leader = D[0]
    lnx, lny = noc(device, *leader)

    # ---- gate/up cores
    gu_cp_rt, gu_wr_rt, gu_rd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(cfg.GR):
        for j in range(cfg.GC):
            x, y = cfg.grid_x0 + i, j
            ps = p.pieces[i]
            prev = noc(device, x - 1, y) if i > 0 else (0, 0)
            nxt = noc(device, x + 1, y) if i + 1 < cfg.GR else (0, 0)
            cp = [st.buffer_address(), len(ps)]
            wr = [len(ps), j, prev[0], prev[1]]
            for r, k0, k1, fl, xoff in ps:
                cp += [k0, k1, fl, xoff]
                wr += [fl, p.qpos.get(r, 0)]
            gu_cp_rt[x][y] = cp
            gu_wr_rt[x][y] = wr
            has_pre = int(ps[-1][3] & 1)
            gu_rd_rt[x][y] = [x_t.buffer_address(), w_t.buffer_address(), i, has_pre, nxt[0], nxt[1], j]
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
    gu_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    gu_cfg.unpack_to_dest_mode = utd
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kpath("gu_compute.cpp"),
            core_ranges=gset,
            compile_time_args=[KT, NPAIR, p.XT, cfg.nexp],
            named_compile_time_args=[("expert_activation", 0)],
            defines=act_defs,
            runtime_args=gu_cp_rt,
            config=gu_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("gu_writer.cpp"),
            core_ranges=gset,
            compile_time_args=[NPAIR, cfg.R, cfg.NS, KTD, 0, cfg.GC, cfg.nexp]
            + [hring.buffer_address(), flags.buffer_address(), pin.buffer_address()]
            + drect
            + [len(D), 0, 0, htile, st.buffer_address()],
            defines=[("WS_NO_SEND", "1")] if cfg.debug_no_send else [],
            runtime_args=gu_wr_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("gu_reader.cpp"),
            core_ranges=gset,
            compile_time_args=[KT, GW, p.XT, cfg.GC * GW, 0, cfg.nexp]
            + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(w_t).get_compile_time_args()),
            runtime_args=gu_rd_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
    ]
    # ---- down cores
    dn_cp_rt, dn_wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    dn_rd_lead, dn_rd_rest = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()  # (one RuntimeArgs per kernel: all its cores)
    for c, (x, y) in enumerate(D):
        dn_cp_rt[x][y] = [st.buffer_address()]
        (dn_rd_lead if c == 0 else dn_rd_rest)[x][y] = [wd_t.buffer_address(), c * CW, st.buffer_address()]
        dn_wr_rt[x][y] = [out_t.buffer_address(), c * CW] + list(p.order)
    dn_nexp = 0 if cfg.debug_no_send else cfg.nexp
    rd_ct = (
        [KTD, CW, cfg.HT, cfg.R, cfg.NS, cfg.GC, dn_nexp, flags.buffer_address(), lnx, lny, len(D)]
        + [len(grects)]
        + [v for r in (grects + [(0, 0, 0, 0, 0)])[:2] for v in r]
        + list(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args())
    )
    lead_set, rest_set = crs([leader]), crs(D[1:])
    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)
    kernels += [
        ttnn.KernelDescriptor(
            kernel_source=kpath("dn_compute.cpp"),
            core_ranges=dset,
            compile_time_args=[KTD, CW, cfg.R, dn_nexp],
            runtime_args=dn_cp_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("dn_reader.cpp"),
            core_ranges=lead_set,
            compile_time_args=rd_ct,
            defines=[("LEADER", "1")],
            runtime_args=dn_rd_lead,
            config=dm0,
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("dn_reader.cpp"),
            core_ranges=rest_set,
            compile_time_args=rd_ct,
            runtime_args=dn_rd_rest,
            config=dm0,
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("dn_writer.cpp"),
            core_ranges=dset,
            compile_time_args=[CW, cfg.HT, cfg.R, dn_nexp] + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args()),
            runtime_args=dn_wr_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
    ]
    cbs = [
        # gate/up
        cb(0, cfg.x_dt, p.XT, gset),
        cb(1, cfg.w_dt, KT * GW, gset),
        cb(4, ttnn.float32, GW, gset),
        ttnn.cb_descriptor_from_sharded_tensor(5, pin),
        cb(16, cfg.h_dt, 2 * NPAIR, gset),
        # down
        cb(1, cfg.w_dt, KTD * CW, dset),
        ttnn.cb_descriptor_from_sharded_tensor(2, hring),
        cb(16, cfg.out_dt, 4 * CW, dset),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=aset, initial_value=0) for s in range(5)]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [x_t, w_t, wd_t, hring, flags, pin, st, out_t]
    return prog, io, st


def interleave_gu(wg, wu):
    """[K, I] gate and up -> [K, 2 I] with 32-column tiles g0 u0 g1 u1 .."""
    K, I = wg.shape
    return torch.stack([wg.reshape(K, I // 32, 32), wu.reshape(K, I // 32, 32)], dim=2).reshape(K, 2 * I)
