# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT fused attention helpers (``ttnn.generic_op``), see tt/attn_kernels/*.cpp.  Env flags (each default off until gated):
DSV41_ATTN_FUSED_ROPE: partial RoPE as one program (rotation-matrix matmul per 32-col tile) in place of mul + 512x512 matmul + addcmul."""

import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_kernels")


def flag(name, default="0"):
    return os.environ.get(name, default) == "1"


def _kernel(name, core_set, ct, rt, cfg, common):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common,
        config=cfg,
    )


def rope_inplace(x, c, s, users, rows_layout=False):
    """Rotate dims 448..511 of x IN PLACE (adjacent pairs).  x bf16 TILE DRAM: heads layout [1,T,32,512] (page = t*16+n; tables
    [1,T,1,512] row 0 of page t*16+n) or rows layout [1,1,T,512] (page = n; every user shares table row 0 of tables [1,1,B,512]).
    ``users`` = T (heads) or 1 (rows).  c, s: bf16 TILE tables (s may be the negated table for the inverse rotation)."""
    assert x.dtype == ttnn.bfloat16 and c.dtype == ttnn.bfloat16 and s.dtype == ttnn.bfloat16
    mesh = x.device()
    XS = 0 if rows_layout else 16
    n = users * 2
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n]
    assert len(cores) == n
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for w, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [w]
    bf = ttnn.bfloat16
    X, R, SC, O = range(4)
    cbs = [
        _cb(core_set, X, 1, 2048, bf),
        _cb(core_set, R, 1, 2048, bf),
        _cb(core_set, SC, 1, 256, bf),
        _cb(core_set, O, 1, 2048, bf),
    ]
    TS = XS
    reader = _kernel(
        "rope_reader.cpp",
        core_set,
        [X, R, SC, XS, TS] + _acc(x) + _acc(c) + _acc(s),
        rt,
        ttnn.ReaderConfigDescriptor(),
        [x.buffer_address(), c.buffer_address(), s.buffer_address()],
    )
    writer = _kernel(
        "rope_writer.cpp", core_set, [O, XS] + _acc(x), rt, ttnn.WriterConfigDescriptor(), [x.buffer_address()]
    )
    compute = _kernel(
        "rope_compute.cpp",
        core_set,
        [X, R, O],
        ttnn.RuntimeArgs(),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0xA70, users, XS, tuple(_acc(x)), tuple(_acc(c)), tuple(_acc(s)))
    ttnn.generic_op([c, s, x], prog)
    return x
