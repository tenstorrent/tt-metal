# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58724 third review): exhaustive bit dump of the dest-reuse forms on tiles other than 32x32, per-tile
hand-off against per-face: 16x32 (two full faces, deepseek_v3_b1's RMSNorm at widths 1536 and 512), 16x16 (one face,
EltwiseMul in moe_routed_expert and the dram streaming matmuls; per-face on both sides under the one-face rule), 8x32 (two 8-row
faces, GatedReduce's SRAM path; per-face on both sides) and 32x32 (ReduceToOneB1's add, RMSNorm at 7168).

The same program as test_eb_dump_reuse.py (kernels/eb_dump_reuse.cpp on 64 cores: DEST <- a tile by copy_tile, then the dest-reuse
op with the L1 tile b), with the CBs and tensors in the tile shape. EB_DUMP_PER_TILE (0 per-face: main's program; 1 per-tile) sets
ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE. DEST values: every bf16 pattern; b: the bf16 special and normal set."""
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
FORMS = {"srca": 0, "srcb": 1}
TILES = {"t16x32": (16, 32), "t16x16": (16, 16), "t8x32": (8, 32), "t32x32": (32, 32)}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NX - 1, NY - 1))])


def _cb(cb_id, dtype, grid, tile):
    t = ttnn.Tile(list(tile))
    page = t.get_tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(t))
    return ttnn.CBDescriptor(total_size=2 * page, core_ranges=grid, format_descriptors=[fmt])


def _run(device, ta, tb, out_dt, n, op, form, fid, fp32, per_tile, tile):
    grid = _grid()
    spec = ttnn.TensorSpec(list(ta.shape), out_dt, ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM, tile=ttnn.Tile(list(tile)))
    tout = ttnn.allocate_tensor_on_device(spec, device)
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
    cbs = [_cb(0, ttnn.bfloat16, grid, tile), _cb(1, ttnn.bfloat16, grid, tile), _cb(16, out_dt, grid, tile)]
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    device.clear_program_cache()
    return out_bits(ttnn.generic_op([ta, tb, tout], prog))


def _data(w):
    """a (DEST) and b (L1) as (rows, w) bf16 bits; per-element b operand values."""
    B = b16_set()
    nb = B.size
    i = np.tile(np.arange(65536), nb)
    j = np.repeat(np.arange(nb), 65536)
    a, b = PATS[i], B[(i + j) % nb]
    return a.reshape(-1, w), b.reshape(-1, w), f32_of_bf16(b)


_CACHE = {}


def _tensors(device, tile):
    if tile not in _CACHE:
        _CACHE.clear()
        _CACHE[tile] = _make_tensors(device, tile)
    return _CACHE[tile]


def _make_tensors(device, tile):
    h, w = tile
    a, b, bval = _data(w)
    rows = a.shape[0]
    ntiles = rows // h
    n = ntiles // (NX * NY)
    assert n * NX * NY == ntiles and ntiles * h == rows
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([h, w]), device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(1, 1, rows, w), **kw)
    tb = ttnn.from_torch(bf16_from_bits(b.reshape(-1)).reshape(1, 1, rows, w), **kw)
    return a, b, bval, ta, tb, n, ntiles


CFG = [(t, op, form, "LoFi", dest) for t in TILES for op in ("add", "sub") for form in FORMS for dest in DEST]
CFG += [(t, "mul", form, fid, dest) for t in TILES for form in FORMS for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4") for dest in DEST]


@pytest.mark.parametrize("tname, op, form, fid, dest", CFG, ids=["-".join(c) for c in CFG])
def test_reuse_t(device, tname, op, form, fid, dest):
    t0 = time.time()
    tile = TILES[tname]
    a, b, bval, ta, tb, n, ntiles = _tensors(device, tile)
    fp32, out_dt = DEST[dest]
    pf = _run(device, ta, tb, out_dt, n, op, form, fid, fp32, False, tile)
    pt = _run(device, ta, tb, out_dt, n, op, form, fid, fp32, True, tile)
    av = f32_of_bf16(a.reshape(-1))
    # DEST_TO_SRCA: DEST (a) is SrcA, the result is a op b; DEST_TO_SRCB: DEST is SrcB, the result is b op a
    x, y = (av, bval) if form != "srcb" else (bval, av)
    d = Diff(f"reuse_{tname}_{op}_{form}_{fid}_{dest} [per-face | per-tile]", op, ("PF", "PT"))
    d.add(pf, pt, x, y)
    d.report(f"({ntiles} {tile[0]}x{tile[1]} tiles, every bf16 pattern in DEST against {b16_set().size} b values, {time.time() - t0:.1f} s)")


@pytest.mark.parametrize("tname", list(TILES))
def test_control_t(device, tname):
    """LoFi against HiFi4 multiply, both per-tile: must differ (the in-process switch works)."""
    tile = TILES[tname]
    a, b, bval, ta, tb, n, ntiles = _tensors(device, tile)
    lo = _run(device, ta, tb, ttnn.bfloat16, n, "mul", "srca", "LoFi", False, True, tile)
    hi = _run(device, ta, tb, ttnn.bfloat16, n, "mul", "srca", "HiFi4", False, True, tile)
    d = Diff(f"reuse control {tname} mul srca d16 [LoFi | HiFi4]", "mul", ("LoFi", "HiFi4"))
    d.add(lo, hi, f32_of_bf16(a.reshape(-1)), bval)
    d.report()
