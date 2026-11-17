# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer's head as a router matmul plus one routing program (kernels wsr_*.cpp), every chip on its own
tokens: logits = x Wr (ttnn matmul, fp32 out, HiFi4), then on every core of the chip, for a contiguous range of row
tiles: the top-K experts of each token (BRISC, NOC0 reads of the logit tiles, integer compares of the fp32 bits),
exp of the picks minus the largest (compute, fp32), the gate weights e_k / sum e and records (t, k, e) (NCRISC), and
the routing outputs through an exchange in L1: the records to the experts' owner cores (counts, offsets, runs), each
owner's per-expert token lists and result-row lists (t S + k) written to their pages, the counts gathered on core 0.
DRAM reads on NOC0, DRAM writes posted on NOC1 (fenced), no multicasts, no DRAM round trip inside the program."""

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_end import end_cores, token_ranges
from models.demos.ws_expert.ws_expert import noc, sharded_l1

CAP = 16384  # records an owner core takes (its experts' routed rows, run padding included)


def head_outputs(mesh, lc):
    """The routing program's outputs: token lists / destination lists [nexp, RS], counts [1, nexp] uint32, w [TX, 16]
    fp32 (row-major), and the fence tensor (NUM_DRAM_BANKS 64-B pages, one a bank)."""
    rm, dram = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    e = lambda shape, dt: ttnn.empty(shape, dtype=dt, layout=rm, device=mesh, memory_config=dram)
    return dict(
        idx_t=e([lc.nexp, lc.RS], ttnn.uint32),
        dest_t=e([lc.nexp, lc.RS], ttnn.uint32),
        counts_t=e([1, lc.nexp], ttnn.uint32),
        w=e([lc.TX, 16], ttnn.float32),
        fence_t=e([16, 16], ttnn.uint32),
    )


def dm(risc):
    """BRISC on NOC0 (0) or NCRISC on NOC1 (1)."""
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_0 if risc == 0 else ttnn.NOC.NOC_1,
    )


def mesh_program(mesh, p):
    R, Cn = tuple(mesh.shape)
    mpd = ttnn.MeshProgramDescriptor()
    mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, Cn - 1))] = p
    return mpd


def build_route(mesh, lc, logits_t, outs):
    """generic_op mesh program of the routing; returns (mesh program, io tensors, stamps)."""
    K, NE, S, T = lc.K, lc.nexp, lc.S, lc.T
    NET = NE // 32
    cores = end_cores(mesh)
    NC = len(cores)
    EPO = -(-NE // NC)
    NOWN = -(-NE // EPO)
    cset = crs(cores)
    rng = token_ranges(lc.TX // 32, NC)
    rcsrc = max(n for _, n in rng) * 32 * K
    w16 = lambda words: sharded_l1(mesh, cores, [-(-words // 16), 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    lg = sharded_l1(mesh, cores, [NET * 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)  # (DRAM reads: 64 B aligned)
    bufs = dict(
        wst=w16(32 * 16),
        rec=w16(rcsrc),
        srt=w16(rcsrc + 4 * NOWN),
        cnt_in=w16(NC),
        off_in=w16(NOWN),
        recbuf=w16(CAP),
        la=w16(CAP + 32 * EPO),
        lb=w16(CAP + 32 * EPO),
        cnt_all=w16(NE),
        aux=w16(3 * 256),
    )
    for t in bufs.values():
        assert t.buffer_address() % 64 == 0
    st = stamps_tensor(mesh, cores)
    xy = [noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8 for c in cores]
    sel_rt, exp_rt, rt_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for me, ((x, y), (r0, n)) in enumerate(zip(cores, rng)):
        sel_rt[x][y] = [logits_t.buffer_address(), r0, n, T, st.buffer_address()]
        exp_rt[x][y] = [n]
        rt_rt[x][y] = [me, r0, n, T] + [outs[k].buffer_address() for k in ("w", "idx_t", "dest_t", "counts_t")] + [
            outs["fence_t"].buffer_address(),
            st.buffer_address(),
        ] + xy
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg.unpack_to_dest_mode = utd
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsr_select.cpp"),
            core_ranges=cset,
            compile_time_args=[NET, K] + ta(logits_t),
            runtime_args=sel_rt,
            config=dm(0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsr_exp.cpp"), core_ranges=cset, runtime_args=exp_rt, config=ccfg
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsr_route.cpp"),
            core_ranges=cset,
            compile_time_args=[K, NE, S, EPO, NOWN, NC, CAP, lc.RS]
            + [bufs[k].buffer_address() for k in ("wst", "rec", "srt", "cnt_in", "off_in", "recbuf", "la", "lb")]
            + [bufs["cnt_all"].buffer_address(), bufs["aux"].buffer_address()]
            + ta(outs["w"])
            + ta(outs["idx_t"])
            + ta(outs["dest_t"])
            + ta(outs["counts_t"])
            + ta(outs["fence_t"]),
            runtime_args=rt_rt,
            config=dm(1),
        ),
    ]
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, lg),
        cb(1, ttnn.float32, 2, cset),
        cb(2, ttnn.float32, 2, cset),
        cb(3, ttnn.uint16, 2, cset, page=32 * K * 2),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(2)]
    mpd = mesh_program(mesh, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    io = [logits_t, lg, st] + list(bufs.values()) + [outs[k] for k in ("w", "idx_t", "dest_t", "counts_t", "fence_t")]
    return mpd, io, st


def route_span_us(st, ci, clk):
    """The routing program's stamps on chip ci, us from its earliest start: the latest picks done, records done (A),
    exchange done (B), lists done (C) and end over the cores; and the earliest start as a raw wall-clock value."""
    s = ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF
    s = s.reshape(-1, 8 * 64)
    s0 = int(s[:, 0].min())
    rel = lambda v: ((v - s0 + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    out = {k: int(rel(s[:, i]).max()) / clk for k, i in (("select", 1), ("records", 2), ("exchange", 3), ("lists", 4))}
    out["end"] = int(rel(s[:, 63]).max()) / clk
    return out, s0


def clock_programs(mesh, n):
    """n one-core programs, program i writing the chip's wall clock to word i of a stamps tensor; returns (mesh
    programs, the stamps tensor)."""
    core = (0, 0)
    st = stamps_tensor(mesh, [core])
    progs = []
    for i in range(n):
        rt = ttnn.RuntimeArgs()
        rt[core[0]][core[1]] = [st.buffer_address(), i]
        k = ttnn.KernelDescriptor(
            kernel_source=kpath("wsr_clock.cpp"), core_ranges=crs([core]), runtime_args=rt, config=dm(0)
        )
        progs.append(mesh_program(mesh, ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])))
    return progs, st


def clock_stamps(st, ci, n):
    """The n clock words of chip ci (raw wall-clock values)."""
    return [int(v) for v in (ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF).flatten()[:n]]


def router_logits(mesh, x_t, wr_t, out=None):
    """logits = x Wr: [TX, nexp] fp32 TILE (HiFi4, fp32 accumulation), into out when given."""
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    return ttnn.matmul(x_t, wr_t, dtype=ttnn.float32, compute_kernel_config=ck, optional_output_tensor=out)


def top_from_lists(lc, idx, dest, counts):
    """The device's routing as [T, K] expert ids, recovered from its destination lists (each (t, k) exactly once;
    None if not)."""
    top = torch.full((lc.T, lc.K), -1, dtype=torch.long)
    for e in range(lc.nexp):
        d = dest[e, : counts[e]]
        t, k = d // lc.S, d % lc.S
        if bool((top[t, k] != -1).any()):
            return None
        top[t, k] = e
    return None if bool((top < 0).any()) else top
