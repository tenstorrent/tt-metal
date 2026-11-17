# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End pass of the MoE layer (kernels we_*.cpp): y[t] = sum over k < K of w[t, k] rows[t S + k] + rows[t S + K] (the
shared expert's row), S = K + 1, with rows the row-major result buffer [T S, H] bf16 the stream and the head write and
w [T, 16] fp32 the gate weights. Every core of the chip takes a contiguous range of tokens; the row-major y [T, H]
bf16 is what the stream's x reader takes. One program for the mesh: each chip on its own buffers."""

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import sharded_l1

TPR = 4  # tile slots a row (3840 = 3.75 x 1024 elements)


def end_cores(mesh):
    g = mesh.compute_with_storage_grid_size()
    return [(x, y) for y in range(g.y) for x in range(g.x)]


def token_ranges(T, C):
    n = [T // C + (1 if i < T % C else 0) for i in range(C)]
    starts = [sum(n[:i]) for i in range(C)]
    return list(zip(starts, n))


def build_end(mesh, T, K, H, rows_t, w_t, y_t, nbuf=4, fidelity=ttnn.MathFidelity.HiFi4, read_noc=0, cores=None):
    """generic_op mesh program of the end pass (nbuf: tokens of reads in flight a core; read_noc: the NoC of every
    core's reads, 0 or 1, the writes on the other; or "split": half the cores read on NOC0 and write on NOC1, the
    other half the other way round, which puts DRAM reads and acknowledged DRAM writes on both NoCs - a mix that can
    hang Blackhole - so not the default); cores: the cores (default every core); returns (mesh program, io tensors,
    stamps)."""
    S = K + 1
    assert H * 2 <= TPR * 2048 and K <= 16
    cores = cores or end_cores(mesh)
    cset = crs(cores)
    # DRAM <-> L1 transfers need the L1 address congruent to the DRAM one mod 64: the rows and y CBs on 64 B aligned
    # L1 tensors, the w page in a 64 B aligned spot of a scratch tensor
    rows_l1 = sharded_l1(mesh, cores, [nbuf * S * TPR * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out_l1 = sharded_l1(mesh, cores, [2 * TPR * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    w_scr = sharded_l1(mesh, cores, [1, 16 * nbuf + 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    assert rows_l1.buffer_address() % 64 == 0 and out_l1.buffer_address() % 64 == 0
    w_l1 = (w_scr.buffer_address() + 63) // 64 * 64
    st = stamps_tensor(mesh, cores)
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1,
    )
    if read_noc == "split":
        sets = [[c for c in cores if (c[0] + c[1]) % 2 == p] for p in (0, 1)]
    else:
        sets = [cores, []] if int(read_noc) == 0 else [[], cores]
    rng = dict(zip(cores, token_ranges(T, len(cores))))
    cp_rt = ttnn.RuntimeArgs()
    for (x, y), (t0, n) in rng.items():
        cp_rt[x][y] = [n, st.buffer_address()]
    rd_ct = [K, S, H * 2, w_l1, nbuf, 2 * S * 2048] + list(ttnn.TensorAccessorArgs(rows_t).get_compile_time_args())
    rd_ct += list(ttnn.TensorAccessorArgs(w_t).get_compile_time_args())
    wr_ct = [H * 2] + list(ttnn.TensorAccessorArgs(y_t).get_compile_time_args())
    kernels = []
    for noc, cs in enumerate(sets):  # (cores reading on NOC noc, writing on the other)
        if not cs:
            continue
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for x, y in cs:
            t0, n = rng[(x, y)]
            rd_rt[x][y] = [rows_t.buffer_address(), w_t.buffer_address(), t0, n, st.buffer_address()]
            wr_rt[x][y] = [y_t.buffer_address(), t0, n, st.buffer_address()]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("we_reader.cpp"),
                core_ranges=crs(cs),
                compile_time_args=rd_ct,
                runtime_args=rd_rt,
                config=dm(1, noc),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("we_writer.cpp"),
                core_ranges=crs(cs),
                compile_time_args=wr_ct,
                runtime_args=wr_rt,
                config=dm(0, 1 - noc),
            )
        )
    kernels += [
        ttnn.KernelDescriptor(
            kernel_source=kpath("we_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[S],
            runtime_args=cp_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
        ),
    ]
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, rows_l1),
        cb(1, ttnn.bfloat16, 2 * S, cset),
        ttnn.cb_descriptor_from_sharded_tensor(16, out_l1),
    ]
    p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    R, Cn = tuple(mesh.shape)
    mpd = ttnn.MeshProgramDescriptor()
    mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, Cn - 1))] = p
    io = [rows_t, w_t, rows_l1, out_l1, w_scr, st, y_t]
    return mpd, io, st


def end_waits_k(st, ci):
    """Mean over the cores of the end pass's wait counters (K cycles): compute inputs / output space, reader reads /
    CB space, writer rows / writes leaving."""
    s = (ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 8 * 64).double()
    names = ("cp_in", "cp_out", "rd_reads", "rd_space", "wr_rows", "wr_flush")
    return {n: round(float(s[:, i + 1].mean()) / 1000, 1) for i, n in enumerate(names)}


def end_span_us(st, ci, clk):
    """The end pass's device time on chip ci: earliest compute start to latest compute end over the cores."""
    s = ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF
    s = s.reshape(-1, 8 * 64)
    rel = lambda v: ((v - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return (int(rel(s[:, 63]).max()) - int(rel(s[:, 0]).min())) / clk
