# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: several FabricStream RS phases in ONE program (the PMK layer's P8 and P15 on the same lanes and reducers),
composed the way the PMK composite does it (tt/pf_stream.composite_source: every phase body in its own namespace with
PH_CT0 / ph_rt0, includes hoisted): per (role, RISC) one kernel running the phases in order; the lanes' fabric args
after ALL phases (fab_at = the total: every phase opens / closes the same connection); each phase its own sync tensor,
landing buffer and y. The instances must share lanes, reducers, nb (CB pages) and hidden."""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PF


def _rt_dict(rt: ttnn.RuntimeArgs, cores) -> dict:
    out = {}
    for c in cores:
        try:
            out[c] = [int(v) for v in rt[c[0]][c[1]]]
        except IndexError:  # a kernel without runtime args (the compute)
            out[c] = []
    return out


def rsr_n_program(rsrs: list, pts: list, r: int, c: int) -> ttnn.ProgramDescriptor:
    """FabricStreamRSR instances (lanes / reducers mode, DRAM y) as consecutive phases of one program on chip (r, c)"""
    kernels, cbs = rsr_n_parts(rsrs, pts, r, c)
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    rsrs[0].connect(prog, r, c, 0)  # the lanes' send composite: the fabric args after all phases
    return prog


def rsr_n_parts(rsrs: list, pts: list, r: int, c: int, ready: bool = False):
    """(kernels [lanes send, lanes read, reducers read, reduce, write], cbs) of the RS phases; the caller connect()s the
    lanes' send kernel (its index in the final program) via rsrs[0].connect"""
    r0 = rsrs[0]
    assert all(
        x.lanes == r0.lanes and x.reducers == r0.reducers and x.nb == r0.nb and x.hidden == r0.hidden for x in rsrs
    )
    assert r0.reducers and not r0.members, "the reducers + DRAM y form"
    # pass 1: the send lists' lengths (fab_at = their total); pass 2: the kernels with fab_at set
    builds = [x._build(p, r, c, ready=ready, fab_at=0) for x, p in zip(rsrs, pts)]
    send_len = [len(_rt_dict(b[0][0].runtime_args, r0.lanes)[r0.lanes[0]]) for b in builds]
    builds = [x._build(p, r, c, ready=ready, fab_at=sum(send_len)) for x, p in zip(rsrs, pts)]
    kernels, cbs = [], builds[0][1]
    n_k = len(builds[0][0])
    for ki in range(n_k):
        ks = [b[0][ki] for b in builds]
        cores = r0.lanes if ki < 2 else r0.reducers
        risc = "c" if ki == 3 else "dm"  # [rs_send, rs_read, red_read, rs_reduce, red_write]
        rts = [_rt_dict(k.runtime_args, cores) for k in ks]
        cts = [list(k.compile_time_args) for k in ks]
        cb_ = [sum(len(x) for x in cts[:i]) for i in range(len(cts))]
        lens = [len(next(iter(d.values()))) for d in rts]
        rb_ = [sum(lens[:i]) for i in range(len(rts))]
        for d in rts:  # every core of the role has the same list length within a phase (the composite contract)
            assert len({len(v) for v in d.values()}) == 1
        rt = ttnn.RuntimeArgs()
        for cc in cores:
            rt[cc[0]][cc[1]] = [v for d in rts for v in d[cc]]
        defs = sorted({tuple(dd) for k in ks for dd in k.defines})
        k0 = ks[0]
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=PF.composite_source([k.kernel_source for k in ks], cb_, rb_, risc,
                                              "    compute_kernel_hw_startup(1, 2, 3);" if risc == "c" else ""),  # A, B, Y
            source_type=k0.source_type,
            core_ranges=k0.core_ranges, compile_time_args=[v for x in cts for v in x], runtime_args=rt,
            config=k0.config, defines=[list(dd) for dd in defs]))  # fmt: skip
    return kernels, cbs


def rsr_n_call(rsrs: list, pts: list):
    mesh = rsrs[0].mesh
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(int(mesh.shape[0])):
        for c in range(int(mesh.shape[1])):
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = rsr_n_program(rsrs, pts, r, c)
    ttnn.generic_op(list(pts) + [t for x in rsrs for t in (x.r, x.y, x.sync)], mpd)
    return [x.y for x in rsrs]
