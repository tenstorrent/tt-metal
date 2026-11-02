# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layout moves around the prefill sparse_sdpa (DSV41_PF_RETILE=1, default OFF; cores-for-prefill study).

q_to_pad(x, user, qpad): x [U, H, C, 512] bf16 TILE -> the H real heads of ``user`` as row-major sticks into the persistent zero-padded qpad [1, 32, C, 512] (replaces slice + to_layout + slice_write).
o_from_pad(o, out, user): o [1, 32, C, 512] bf16 row-major (sparse_sdpa output) -> its first H heads tilized into out [U, H, C, 512] TILE at ``user`` (replaces slice + to_layout + concat).
Each is one generic_op over up to 120 cores (a unit = one (head, 32-token tile row))."""
import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pf_kernels")
bf16 = ttnn.bfloat16


def _kernel(name, core_set, ct, rt, cfg, common=None):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common or [],
        config=cfg,
    )


def _run(mode, src, dst, user, H, C, units_cap=120):
    ROWT = C // 32
    units = H * ROWT
    mesh = src.device()
    g = mesh.compute_with_storage_grid_size()
    n_cores = min(units_cap, units, int(g.x) * int(g.y))
    cores = [(cx, cy) for cy in range(int(g.y)) for cx in range(int(g.x))][:n_cores]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    base, extra = divmod(units, n_cores)
    rt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for i, (cx, cy) in enumerate(cores):
        nu = base + (1 if i < extra else 0)
        rt[cx][cy] = [u0, nu, user]
        crt[cx][cy] = [nu]
        wrt[cx][cy] = [u0, nu, user]
        u0 += nu
    CIN, COUT = 0, 1
    in_pages, in_page = (16, 2048) if mode == 0 else (32, 1024)
    cbs = [_cb(core_set, CIN, 2 * in_pages, in_page, bf16), _cb(core_set, COUT, 32, 2048, bf16)]
    reader = _kernel(
        "retile_reader.cpp",
        core_set,
        [CIN, mode, ROWT, H, C] + _acc(src),
        rt,
        ttnn.ReaderConfigDescriptor(),
        common=[src.buffer_address()],
    )
    compute = _kernel(
        "retile_compute.cpp",
        core_set,
        [CIN, COUT, mode],
        crt,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
    )
    writer = _kernel(
        "retile_writer.cpp",
        core_set,
        [COUT, mode, ROWT, H, C] + _acc(dst),
        wrt,
        ttnn.WriterConfigDescriptor(),
        common=[dst.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0x6D1 + mode, H, C, user, tuple(_acc(src)), tuple(_acc(dst)))
    ttnn.generic_op([src, dst], prog)
    return dst


def q_to_pad(x, user, qpad):
    U, H, C, W = (int(v) for v in x.shape)
    assert W == 512 and x.dtype == bf16 and x.layout == ttnn.TILE_LAYOUT and qpad.layout == ttnn.ROW_MAJOR_LAYOUT
    assert int(qpad.shape[1]) >= H and int(qpad.shape[2]) == C
    return _run(0, x, qpad, user, H, C)


def o_from_pad(o, out, user):
    U, H, C, W = (int(v) for v in out.shape)
    assert W == 512 and o.dtype == bf16 and o.layout == ttnn.ROW_MAJOR_LAYOUT and out.layout == ttnn.TILE_LAYOUT
    return _run(1, o, out, user, H, C)


_CAT = {}


def cat_heads(o, max_cores=120):
    """o [U, H, C, 512] bf16 TILE -> [1, 1, U * C, (H + 1) * 512]: a zero head group first, then the H heads (what nlp_concat_heads + concat([zero, c], dim=3) + reshape produced).
    One persistent buffer per (device, U * C): its zero columns are never written, the head columns are overwritten on every call (valid while the consumer of the previous call is done:
    the layers run one after the other on one queue)."""
    U, H, C, W = (int(v) for v in o.shape)
    assert W == 512 and o.dtype == bf16 and o.layout == ttnn.TILE_LAYOUT
    mesh = o.device()
    key = (id(mesh), U, H, C)
    if key not in _CAT:
        _CAT[key] = ttnn.zeros(
            [1, 1, U * C, (H + 1) * 512],
            dtype=bf16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    out = _CAT[key]
    ROWT = C // 32
    units = U * H * ROWT
    g = mesh.compute_with_storage_grid_size()
    n_cores = min(max_cores, units, int(g.x) * int(g.y))
    cores = [(cx, cy) for cy in range(int(g.y)) for cx in range(int(g.x))][:n_cores]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    base, extra = divmod(units, n_cores)
    rt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for i, (cx, cy) in enumerate(cores):
        nu = base + (1 if i < extra else 0)
        rt[cx][cy] = [u0, nu]
        wrt[cx][cy] = [u0, nu]
        u0 += nu
    cbs = [_cb(core_set, 0, 32, 2048, bf16)]
    reader = _kernel(
        "cat_reader.cpp",
        core_set,
        [0, ROWT, H] + _acc(o),
        rt,
        ttnn.ReaderConfigDescriptor(),
        common=[o.buffer_address()],
    )
    writer = _kernel(
        "cat_writer.cpp",
        core_set,
        [0, ROWT, H] + _acc(out),
        wrt,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0x6D9, U, H, C, tuple(_acc(o)), tuple(_acc(out)))
    ttnn.generic_op([o, out], prog)
    return out
