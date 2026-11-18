# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B49 (HY4_B49_RSTAIL=1, default OFF): the tail of the exact TP32 prefill MoE block as ONE program: y3[t] = bf16(((yg_0[r n + t] + yg_1[r n + t]) + yg_2[...]) + yg_3[...]) for the own n tokens of this
mesh row r, from the all-gathered fp32 row partials yg [1,1,4 Tt,Dc] (replaces slice x4, add x3, typecast, to_layout, reshape, embedding, reshape: 13 ops)."""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_pf import _acc, _cb, _cores, _kernel, _rt, _unpack_modes, bf16, f32

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ihc_pf_kernels")


def enabled():
    return os.environ.get("HY4_B49_RSTAIL", "0") == "1"


def make_rowbase(md, n, rows_axis_index=None):
    """per-device uint32 [1,1,1,32] whose word 0 = (mesh row r) * n / 32; sharded over the mesh rows."""
    R = int(md.shape[0])
    t = torch.zeros(R, 1, 1, 32, dtype=torch.int32)
    for r in range(R):
        t[r, 0, 0, 0] = r * (n // 32)
    return upload(md, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0 if R > 1 else None)


def rs_tail(md, yg, rowbase, n, Dc):
    """yg fp32 TILE [1,1,4 Tt,Dc] (Tt = 4 n), rowbase from make_rowbase -> y3 bf16 TILE [1,1,n,Dc]."""
    TR, NCOL = n // 32, Dc // 32
    BLK = (4 * n) // 32
    assert tuple(yg.shape) == (1, 1, 4 * 4 * n, Dc) and yg.dtype == f32
    grid = md.compute_with_storage_grid_size()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n, Dc]), bf16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    cs, co = _cores(grid, TR)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    cfg.unpack_to_dest_mode = _unpack_modes(0)
    rd = _kernel(
        "rs_reader.cpp",
        cs,
        [0, 1, BLK, NCOL] + _acc(yg) + _acc(rowbase),
        _rt(co, lambda i: [i]),
        ttnn.ReaderConfigDescriptor(),
        [yg.buffer_address(), rowbase.buffer_address()],
    )
    wr = _kernel(
        "w_generic.cpp",
        cs,
        [2, 2048, NCOL, 1, 0, 4, 0xFFFF, 0, 0] + _acc(out) + _acc(out),
        _rt(co, lambda i: [i * NCOL, 0]),
        ttnn.WriterConfigDescriptor(),
        [out.buffer_address(), out.buffer_address()],
    )
    cp = _kernel("rs_compute.cpp", cs, [0, 2, NCOL], ttnn.RuntimeArgs(), cfg, [])
    prog = ttnn.ProgramDescriptor(
        kernels=[rd, wr, cp],
        semaphores=[],
        cbs=[_cb(cs, 0, 8, 4096, f32), _cb(cs, 1, 1, 128, f32), _cb(cs, 2, 4, 2048, bf16)],
    )
    prog.custom_program_hash = (0x4B50 << 40) | (hash(("rst", n, Dc)) & ((1 << 40) - 1))
    ttnn.generic_op([yg, rowbase, out], prog)
    return out
