# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58818 review): exhaustive bit dump of the dest-reuse forms, per-tile hand-off against per-face.

A ttnn.generic_op program on 64 cores (kernels/eb_dump_reuse.cpp): DEST <- a tile (copy_tile), then the dest-reuse op with the
L1 tile b: DEST_TO_SRCA, DEST_TO_SRCB (eltwise_binary.h), and bge_m3 balanced layernorm's row-broadcast DEST_TO_SRCA form.
The host define EB_DUMP_PER_TILE (0 per-face: main's program; 1 per-tile) sets ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE.
DEST values: every bf16 pattern; b: the bf16 special and normal set, each pattern against each value."""
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import PATS, Diff, b16_set, bf16_from_bits, f32_of_bf16, kernel_variants, out_bits

READER = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/reconfig/reader_inputs.cpp"
WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
KERNEL = "tests/eb_r3_ci/kernels/eb_dump_reuse.cpp"
NX, NY = 8, 8
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}
DEST = {"d16": (False, ttnn.bfloat16), "d32f": (True, ttnn.float32), "d32b": (True, ttnn.bfloat16)}
OPS = {"add": 0, "sub": 1, "mul": 2}
FORMS = {"srca": 0, "srcb": 1, "rowsrca": 2}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NX - 1, NY - 1))])


def _cb(cb_id, dtype, grid):
    page = 4096 if dtype == ttnn.float32 else 2048
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=2 * page, core_ranges=grid, format_descriptors=[fmt])


def _run(device, ta, tb, out_dt, n, op, form, fid, fp32, per_tile):
    grid = _grid()
    tout = ttnn.allocate_tensor_on_device(ta.shape, out_dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cta = [2] + ttnn.TensorAccessorArgs(ta).get_compile_time_args() + ttnn.TensorAccessorArgs(tb).get_compile_time_args()
    for _ in range(2):
        cta += ttnn.TensorAccessorArgs(ta).get_compile_time_args()
    rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    k = 0
    for y in range(NY):
        for x in range(NX):
            rr[x][y] = [ta.buffer_address(), tb.buffer_address(), n, k * n]
            rw[x][y] = [tout.buffer_address(), n, k * n]
            k += 1
    reader = ttnn.KernelDescriptor(
        kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
        compile_time_args=cta, runtime_args=rr, config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
        compile_time_args=[16] + ttnn.TensorAccessorArgs(tout).get_compile_time_args(), runtime_args=rw,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
        compile_time_args=[n, OPS[op], FORMS[form]], defines=[("EB_DUMP_PER_TILE", "1" if per_tile else "0")],
        runtime_args=[], config=ttnn.ComputeConfigDescriptor(math_fidelity=FID[fid], fp32_dest_acc_en=fp32),
    )
    cbs = [_cb(0, ttnn.bfloat16, grid), _cb(1, ttnn.bfloat16, grid), _cb(16, out_dt, grid)]
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    device.clear_program_cache()
    return out_bits(ttnn.generic_op([ta, tb, tout], prog))


def _data(form):
    """a (DEST) and b (L1) as (rows, 32) bf16 bits, tile t = rows 32t..32t+31; per-element b operand values."""
    B = b16_set()
    nb = B.size
    if form != "rowsrca":
        i = np.tile(np.arange(65536), nb)
        j = np.repeat(np.arange(nb), 65536)
        a, b = PATS[i], B[(i + j) % nb]
        return a.reshape(-1, 32), b.reshape(-1, 32), f32_of_bf16(b)
    T = 64 * nb
    t = np.arange(T)
    jb, u = t // 64, t % 64
    r = np.arange(32)
    c = np.arange(32)
    a = PATS[(u[:, None, None] * 1024 + r[None, :, None] * 32 + c[None, None, :])]
    brow = B[(c[None, :] + jb[:, None] + u[:, None]) % nb]  # (T, 32): row 0 of each b tile
    b = B[(c[None, None, :] + jb[:, None, None] + u[:, None, None] + 7 * r[None, :, None]) % nb]
    b[:, 0, :] = brow
    bval = np.broadcast_to(f32_of_bf16(brow)[:, None, :], (T, 32, 32)).reshape(-1)
    return a.reshape(-1, 32), b.reshape(-1, 32), bval


CFG = [(op, form, fid, dest) for op in ("add", "sub") for form in FORMS for fid in ("LoFi",) for dest in DEST]
CFG += [("mul", form, fid, dest) for form in FORMS for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4") for dest in DEST]


@pytest.mark.parametrize("op, form, fid, dest", CFG, ids=["-".join(c) for c in CFG])
def test_reuse(device, op, form, fid, dest):
    t0 = time.time()
    a, b, bval = _data(form)
    rows = a.shape[0]
    ntiles = rows // 32
    n = ntiles // (NX * NY)
    assert n * NX * NY == ntiles
    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(1, 1, rows, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.from_torch(bf16_from_bits(b.reshape(-1)).reshape(1, 1, rows, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fp32, out_dt = DEST[dest]
    pf = _run(device, ta, tb, out_dt, n, op, form, fid, fp32, False)
    pt = _run(device, ta, tb, out_dt, n, op, form, fid, fp32, True)
    av = f32_of_bf16(a.reshape(-1))
    # DEST_TO_SRCA: DEST (a) is SrcA, the result is a op b; DEST_TO_SRCB: DEST is SrcB, the result is b op a
    x, y = (av, bval) if form != "srcb" else (bval, av)
    d = Diff(f"reuse_{op}_{form}_{fid}_{dest} [per-face | per-tile]", op, ("PF", "PT"))
    d.add(pf, pt, x, y)
    d.report(f"({ntiles} tiles, every bf16 pattern in DEST against {b16_set().size} b values, {time.time() - t0:.1f} s)")


def test_control(device):
    """LoFi against HiFi4 multiply, both per-tile: must differ (the in-process switch works)."""
    a, b, bval = _data("srca")
    rows = a.shape[0]
    n = rows // 32 // (NX * NY)
    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(1, 1, rows, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.from_torch(bf16_from_bits(b.reshape(-1)).reshape(1, 1, rows, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    lo = _run(device, ta, tb, ttnn.bfloat16, n, "mul", "srca", "LoFi", False, True)
    hi = _run(device, ta, tb, ttnn.bfloat16, n, "mul", "srca", "HiFi4", False, True)
    d = Diff("reuse control mul srca d16 [LoFi | HiFi4]", "mul", ("LoFi", "HiFi4"))
    d.add(lo, hi, f32_of_bf16(a.reshape(-1)), bval)
    d.report()
