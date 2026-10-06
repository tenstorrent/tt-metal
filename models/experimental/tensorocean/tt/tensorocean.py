# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Optimized TensorOcean horizontal tracer flux on one Blackhole chip.

One step (TensorOcean.run) = three programs on an 11 x 10 grid of Tensix cores, all on the chip:
  1. kernels/relayout_in.cpp:  natural-layout tracer values, f and mask in DRAM -> the fused kernel's per-core
                               layout (one pass: whole rows read once, reordered in L1, written as blocks)
  2. fused_kernel.py:          the fused flux + accumulation kernel (reader / compute / writer per core)
  3. kernels/relayout_out.cpp: per-core outputs -> natural even, odd outputs [L, N/2, N] in DRAM
Mesh constants (coefficient products, edge signs, 0.5 * dvEdge, 1 / area) are prepared once in prepare().
Usage:
    s = prepare(host_inputs, n, levels, device)
    even, odd = run(s)            # device tensors [levels, n/2, n]; traceable
"""
import os

import torch
import ttnn

from models.experimental.tensorocean.tt import fused_kernel as base
from models.experimental.tensorocean.tt.natural_io import natural_host, upload_natural

PAGE = base.PAGE
KDIR = base.KDIR
NY = 10
# best fused-kernel settings measured at 100 x 100 on a P150: an even 10-level split per core row,
# coefficients fetched one step per DRAM batch, a 2-step multicast ring
TUNED_SMALL = dict(SENDER_LEVELS=10, SB=1, NS=2)


def _arr(name, vals):
    return f"constexpr uint32_t {name}[{len(vals)}] = {{{', '.join(str(int(v)) for v in vals)}}};\n"


def _header(plan):
    s = "namespace R {\n"
    for k, v in dict(
        N=plan.n,
        M=plan.n + 4,
        MH=(plan.n + 4) // 2,
        H=plan.H,
        CELL_LEN=plan.cell_len,
        F_LEN=plan.f_len,
        NBLK=plan.nblk,
        NBX=plan.nbx,
        L=plan.L,
        PAGE=PAGE,
        OUT_LEN=plan.out_len,
    ).items():
        s += f"constexpr uint32_t {k} = {v};\n"
    s += _arr("BAND0", [a for a, _ in plan.bands]) + _arr("BANDW", [b - a for a, b in plan.bands])
    s += _arr("G_ROWS", [g.rows for g in plan.groups]) + _arr("G_COLS", [g.cols for g in plan.groups])
    return s + "}\n"


def _source(hdr, name):
    lines = (KDIR / name).read_text().splitlines()
    inc = [l for l in lines if l.startswith("#include")]
    rest = [l for l in lines if not l.startswith("#include")]
    return "\n".join(inc) + "\n#include <cstdint>\n" + hdr + "\n".join(rest) + "\n"


def _program(src, cores, cbytes, ct, args_r, args_w):
    """One data-movement kernel on both RISCs of every core, each with its own scratch CB (0 and 1)."""
    sc = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    kr = ttnn.KernelDescriptor(
        kernel_source=src,
        source_type=sc,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=args_r,
        config=ttnn.ReaderConfigDescriptor(),
    )
    kw = ttnn.KernelDescriptor(
        kernel_source=src,
        source_type=sc,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=args_w,
        config=ttnn.WriterConfigDescriptor(),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=cbytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=cbytes)],
        )
        for i in (0, 1)
    ]
    return ttnn.ProgramDescriptor(kernels=[kr, kw], semaphores=[], cbs=cbs)


def _jobs_args(nbx, njobs, addrs):
    """Jobs j = 0 .. njobs-1 spread over the reader (j = cid, cid + 2C, ...) and writer (cid + C, ...)."""
    ncores = nbx * NY
    ar, aw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for x in range(nbx):
        for y in range(NY):
            cid = y * nbx + x
            ar[x][y] = [cid, njobs, 2 * ncores, 0] + addrs
            aw[x][y] = [cid + ncores, njobs, 2 * ncores, 1] + addrs
    return ar, aw


def prepare(host, n, levels, device, dtype=ttnn.float32):
    assert dtype == ttnn.float32, "fp32 kernel"
    if n <= 100 and not os.environ.get("TENSOROCEAN_NO_TUNE"):
        for k, v in TUNED_SMALL.items():
            setattr(base, k, v)
    s = base.prepare(host, n, levels, device, dtype)
    plan, t = s["plan"], s["t"]
    # the per-step arrays base.prepare built on the host are cleared: run() rebuilds them on the chip every step
    for k in ("CELL", "FMK"):
        z = ttnn.from_torch(
            torch.zeros(list(t[k].shape)),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(z, t[k])
        ttnn.deallocate(z)
    nat = upload_natural(natural_host(host), device)
    half = n // 2
    outs = [
        ttnn.from_torch(
            torch.zeros(levels, half, n),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    nbx = plan.nbx
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nbx - 1, NY - 1))])
    hdr = _header(plan)
    njobs = 2 * levels  # input: (level, half); output: (level, even/odd part)

    order_in = [
        nat["cell"],
        nat["normalThicknessFlux1"],
        nat["normalThicknessFlux2"],
        nat["advMaskHighOrder1"],
        nat["advMaskHighOrder2"],
        t["CELL"],
        t["FMK"],
    ]
    ct = []
    for x_ in order_in:
        ct += ttnn.TensorAccessorArgs(x_).get_compile_time_args()
    ar, aw = _jobs_args(nbx, njobs, [x_.buffer_address() for x_ in order_in])
    r64 = lambda b: (b + 63) // 64 * 64
    m = n + 4
    rfam_bytes = max((n + 1) * r64((2 * n + 1) * 4), n * r64((n + 1) * 4))
    cbytes_in = 64 + m * r64(m * 4) + rfam_bytes + (2 * plan.cell_len + 12 * plan.f_len) * 4
    prog_in = _program(_source(hdr, "relayout_in.cpp"), cores, cbytes_in, ct, ar, aw)

    order_out = [t["OUT"], outs[0], outs[1]]
    ct = []
    for x_ in order_out:
        ct += ttnn.TensorAccessorArgs(x_).get_compile_time_args()
    ar, aw = _jobs_args(nbx, njobs, [x_.buffer_address() for x_ in order_out])
    cbytes_out = 64 + nbx * plan.out_len * 4 + half * r64(n * 4)
    prog_out = _program(_source(hdr, "relayout_out.cpp"), cores, cbytes_out, ct, ar, aw)

    s.update(nat=nat, outs=outs, prog_in=prog_in, io_in=order_in, prog_out=prog_out, io_out=order_out)
    return s


def run(s):
    """One step: rearrange the natural inputs, run the fused kernel, rearrange the outputs. Returns (even, odd)."""
    ttnn.generic_op(s["io_in"], s["prog_in"])
    base.run(s)
    ttnn.generic_op(s["io_out"], s["prog_out"])
    return tuple(s["outs"])
