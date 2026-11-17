# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The shared expert folded into the end pass: one program, one chip. The shared expert's two clusters take two
rectangles of 4 whole rows (ws_shared, layout "rows"); the end pass runs on the two remaining whole rows (24 cores),
every core reading its routed rows from DRAM.

- End core e of cluster c (the row of end cores the cluster sends to) owns the cluster's row tiles i = e, e + NE, ..
  (global r = c + NCL i), in the order the cluster produces them.
- Each shared down core pack-untilizes its columns of a row tile's 32 rows and writes them, posted on NOC1, half a row
  tile at a time into the owning end core's L1 ring (two slots of 16 rows), then its flag word (posted, the same
  command buffer: it lands after the rows); before reusing a slot it waits for the end core's credit (wsh_reader.cpp
  WSH_FOLD).
- L1: a sharded tensor takes its address range on every core, so the end cores' token rows, y rows and weights are
  plain circular buffers (on the end cores only; the rows first, at the 64-B aligned base, for the DRAM reads) and only
  the ring and small state are tensors.
- The end core reads each token's K routed rows (NOC0), copies the shared row from its ring, multiplies and sums in
  DEST (weights bf16, 1.0 for the shared row; we_compute.cpp), writes y posted on NOC1 with a fence, and returns the
  slot's credit to the writers on NOC0 (wf_reader.cpp, wf_writer.cpp), so nothing enters a cluster's multicast
  rectangle on NOC1 from outside.
The routed rows hold K slots a token (the shared row no longer goes through DRAM)."""

from dataclasses import replace

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_end import TPR
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_shared import SCfg, clusters, shared_parts

CB_ROWS, CB_W, CB_OUT = 20, 21, 22  # (the end pass's CBs, beside the shared expert's)


def fold_cfg(mesh, scfg: SCfg, top=True):
    """The shared expert's config in the fold (rectangles of 4 rows below / above the end rows) and the end cores per
    cluster: the first two rows (top) or the last two."""
    g = mesh.compute_with_storage_grid_size()
    cfg = replace(scfg, layout="rows", row0=2 if top else 0)
    erows = (0, 1) if top else (g.y - 2, g.y - 1)
    return cfg, [[(x, erows[c]) for x in range(g.x)] for c in range(cfg.NCL)]


def end_tokens(c, e, NE, NCL, R, T):
    """End core e of cluster c's tokens."""
    n, k = 0, 0
    while c + NCL * (e + NE * k) < R:
        t0 = (c + NCL * (e + NE * k)) * 32
        n += max(0, min(32, T - t0))
        k += 1
    return n


def build_fold(
    mesh,
    scfg: SCfg,
    K,
    x_t,
    w13_t,
    w2_t,
    rows_t,
    w_t,
    y_t,
    fence_t,
    top=True,
    nbuf=4,
    fidelity=None,
    no_shared=False,
    slices=0,
):
    """generic_op mesh program of the fold; returns (mesh program, io tensors, stamps (shared, its hubs, end cores)).
    rows_t: the routed rows [T K + spare, H] row-major; w_t: the gate weights [TX, 16] fp32; y_t: [T, H] row-major;
    fence_t: NUM_DRAM_BANKS 64-B pages (the y writes' fence); no_shared (a check): the shared rows' weight 0; slices
    (a diagnostic, WF_SLICES): each routed row read as that many equal pieces."""
    cfg, ends = fold_cfg(mesh, scfg, top)
    fidelity = fidelity or ttnn.MathFidelity.HiFi2
    cl = clusters(mesh, cfg)
    active = [a for _, act, _ in cl for a in act]
    ecores = [c for e in ends for c in e]
    NE, NCL, GC, T, H = len(ends[0]), cfg.NCL, cfg.GC, cfg.T, cfg.H
    R, S, rowb = cfg.R, K + 1, H * 2
    ering = sharded_l1(mesh, ecores, [2 * 16, H], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)  # 2 slots of 16 rows
    eflags = sharded_l1(mesh, ecores, [2 * GC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    fcred = sharded_l1(mesh, active, [NE, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    fscr = sharded_l1(mesh, active, [1, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    sh = shared_parts(
        mesh,
        cfg,
        x_t,
        w13_t,
        w2_t,
        rows_t,
        fold=dict(
            ends=ends,
            ring=ering.buffer_address(),
            flags=eflags.buffer_address(),
            fcred=fcred,
            fscr=fscr,
            master=noc(mesh, *ends[0][0])[0] | noc(mesh, *ends[0][0])[1] << 8,
        ),
    )
    w_scr = sharded_l1(mesh, ecores, [1, 16 * nbuf + 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    w_l1 = (w_scr.buffer_address() + 63) // 64 * 64
    est = stamps_tensor(mesh, ecores)
    xy = lambda c: noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # (the start barrier: every writer and end core counts in at the master, end core 0 of cluster 0)
    master = xy(ends[0][0])
    feeders = sh["feeders"]
    members = [xy(a) for a in active + feeders] + [xy(c) for c in ecores]
    for c in range(NCL):
        writers = [xy(a) for a in cl[c][1]]
        for e, (x, y) in enumerate(ends[c]):
            rd_rt[x][y] = [rows_t.buffer_address(), w_t.buffer_address(), c, e, est.buffer_address()] + writers + [
                master
            ] + ([len(members)] + members if (c, e) == (0, 0) else [])
            wr_rt[x][y] = [y_t.buffer_address(), c, e, est.buffer_address(), fence_t.buffer_address()]
            cp_rt[x][y] = [end_tokens(c, e, NE, NCL, R, T), est.buffer_address()]
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    dm = lambda risc, nc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_0 if nc == 0 else ttnn.NOC.NOC_1,
    )
    eset = crs(ecores)
    kernels = sh["kernels"] + [
        ttnn.KernelDescriptor(
            kernel_source=kpath("wf_reader.cpp"),
            core_ranges=eset,
            compile_time_args=[K, rowb, w_l1, nbuf, 2 * S * 2048, NE, NCL, R, T, ering.buffer_address()]
            + [eflags.buffer_address(), GC, fcred.buffer_address(), CB_ROWS, CB_W]
            + ta(rows_t)
            + ta(w_t),
            defines=([("WF_NO_SHARED", "1")] if no_shared else []) + ([("WF_SLICES", str(slices))] if slices else []),
            runtime_args=rd_rt,
            config=dm(1, 0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wf_writer.cpp"),
            core_ranges=eset,
            compile_time_args=[rowb, NE, NCL, R, T, CB_OUT] + ta(y_t) + ta(fence_t),
            runtime_args=wr_rt,
            config=dm(0, 1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("we_compute.cpp"),
            core_ranges=eset,
            compile_time_args=[S],
            defines=[("WE_ROWS_CB", str(CB_ROWS)), ("WE_W_CB", str(CB_W)), ("WE_OUT_CB", str(CB_OUT))],
            runtime_args=cp_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
        ),
    ]
    cbs = sh["cbs"] + [  # (the rows first: the end cores' CB region starts 64 B aligned; sizes multiples of 64)
        cb(CB_ROWS, ttnn.bfloat16, nbuf * S * TPR, eset),
        cb(CB_OUT, ttnn.bfloat16, 2 * TPR, eset),
        cb(CB_W, ttnn.bfloat16, 2 * S, eset),
    ]
    bset = crs(active + feeders + ecores)
    sems = sh["sems"] + [ttnn.SemaphoreDescriptor(id=i, core_ranges=bset, initial_value=0) for i in (2, 3)]
    p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    Rm, Cn = tuple(mesh.shape)
    mpd = ttnn.MeshProgramDescriptor()
    mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(Rm - 1, Cn - 1))] = p
    st, hst = sh["stamps"]
    io = sh["io"][:-1] + [ering, eflags, w_scr, est, w_t, fence_t, rows_t, y_t]
    return mpd, io, (st, hst, est)


def fold_span_us(sts, ci, clk):
    """The fold's device time on chip ci: the earliest compute start (shared or end cores) to the latest compute
    end."""
    s = [ttnn.to_torch(ttnn.get_device_tensors(t)[ci]).to(torch.int64) & 0xFFFFFFFF for t in (sts[0], sts[2])]
    s = [v.reshape(-1, 8 * 64) for v in s]
    base = int(s[0][0, 0])
    rel = lambda v: ((v - base + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    start = min(int(rel(v[:, 0]).min()) for v in s)
    end = max(int(rel(v[:, 63]).max()) for v in s)
    return (end - start) / clk
