# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSV41_CMP_FUSE=1 (default OFF): the data movement of the ratio-2 compressor of the spec verify block as two JIT programs (``ttnn.generic_op``), bit-identical to the composites they replace.

compress_shift(cs, prev_cs, U, n)   cs fp32 TILE [1,1,T,1024] (T = U * n rows, user-major), prev_cs fp32 TILE [1,1,U,1024]  ->  a fp32 TILE [1,1,T,1024],
                                    a[u*n + j] = prev_cs[u] (j = 0) | cs[u*n + j - 1]   (replaces to_layout x3 + concat + slice + reshape: ~100 us -> one program)
commit_select(cs, prev_cs, idx3, U, n)  prev_cs[u] <- cs[u*n + m[u]] in place (m = word 0 of row u of idx3, uint32 [U,1,1024]; idx3 None: m = n - 1)
                                    (replaces gather + to_layout + reshape + copy of the RM cs block: ~100 us per owner layer -> one program)
Both use one core per fp32 tile column (32 columns of the 1024-wide [kv | score] row), sources: tt/attn_kernels/cmp_shift.cpp, cmp_commit.cpp.
"""

import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_fused import _kernel as _k5
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

W = 1024


def enabled():
    return os.environ.get("DSV41_CMP_FUSE", "0") == "1" and os.environ.get("DSV41_SPEC_COMMIT_GATHER", "0") == "1"


def ok(cs, prev_cs, U, n):
    return (
        U * n <= 32
        and cs.dtype == ttnn.float32
        and prev_cs.dtype == ttnn.float32
        and cs.layout == ttnn.TILE_LAYOUT
        and prev_cs.layout == ttnn.TILE_LAYOUT
        and int(cs.shape[-1]) == W
        and int(prev_cs.shape[-1]) == W
    )


def _cores(mesh):
    grid = mesh.compute_with_storage_grid_size()
    return [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][: W // 32]


def _prog(mesh, kname, ct_extra, tens, pages, tag):
    cores = _cores(mesh)
    assert len(cores) == W // 32
    cset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [i]
    accs = sum([_acc(t) for t in tens], [])
    common = [t.buffer_address() for t in tens]
    cb = _cb(cset, 0, pages, 4096, ttnn.float32)
    kd = _k5(kname, cset, [0] + ct_extra + accs, rt, ttnn.ReaderConfigDescriptor(), common)
    prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])
    prog.custom_program_hash = _hash(tag, tuple(ct_extra), tuple(accs))
    return prog


def compress_shift(cs, prev_cs, U, n):
    mesh = cs.device()
    T = U * n
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, W]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    prog = _prog(mesh, "cmp_shift.cpp", [U, n], [cs, prev_cs, out], 3, 0xC51)
    ttnn.generic_op([cs, prev_cs, out], prog)
    return out


def commit_select(cs, prev_cs, idx3, U, n):
    mesh = cs.device()
    mode = 0 if idx3 is None else 1
    ix = idx3 if idx3 is not None else cs  # (dummy accessor slot in mode 0)
    prog = _prog(mesh, "cmp_commit.cpp", [U, n, mode], [cs, ix, prev_cs], 4, 0xC52)
    ttnn.generic_op([cs, prev_cs] if idx3 is None else [cs, ix, prev_cs], prog)
    return prev_cs
