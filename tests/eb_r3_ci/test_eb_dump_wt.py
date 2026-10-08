# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 fourth review): exhaustive bit dump of the whole-tile program (fidelity phase outer) for the
column broadcast multiply of partial-face tiles (1x32 to 8x32) against main's per-face program; the standard form, which keeps
the per-face program on partial faces, is the same-program control.

A ttnn.generic_op program on NX x NY cores (kernels/eb_dump_wt.cpp) streams tiles from DRAM: form 0 the standard op
(mul_tiles, add_tiles, sub_tiles), form 1 mul_tiles_bcast_cols, form 2 sdpa_mul_bcast_col_reuse_tiles (blocks of 8 tiles, P1
from column 0 of the block's first b tile, the second operand and P2 zero). The host define EB_DUMP_PER_TILE sets the kernel's
opt-in: 0 is main's per-face program, 1 the whole-tile program. full: every bf16 pattern (a) against every bf16 pattern (b), in
256 chunks of 2^24 pairs; set: every pattern against the bf16 special and normal set (b16_set)."""
import os
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import PATS, Diff, b16_set, bf16_from_bits, f32_of_bf16, kernel_variants, out_bits

READER = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/reconfig/reader_inputs.cpp"
WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
KERNEL = "tests/eb_r3_ci/kernels/eb_dump_wt.cpp"
NX, NY = 8, 8
CHUNK = 1 << 24
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}
DEST = {"d16": (False, ttnn.bfloat16), "d32": (True, ttnn.float32)}
OPS = {"add": 0, "sub": 1, "mul": 2}
FORMS = {"none": 0, "col": 1, "sdpa": 2}
MAX_CHUNKS = int(os.environ.get("EB_DUMP_MAX_CHUNKS", "0"))


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NX - 1, NY - 1))])


def _cb(cb_id, dtype, tile, pages, grid):
    t = ttnn.Tile(list(tile))
    page = t.get_tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(t))
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid, format_descriptors=[fmt])


def _dev(bits, rows, tile, device):
    return ttnn.from_torch(
        bf16_from_bits(bits.reshape(-1)).reshape(1, 1, rows, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile(list(tile)),
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run(device, ins, tout, n, form, op, fid, fp32, per_tile, tile):
    grid = _grid()
    cta = [len(ins)]
    # reader_inputs.cpp names the accessor args of all four inputs whatever their count: pad with the first input's
    for t in list(ins) + [ins[0]] * (4 - len(ins)):
        cta += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    k = 0
    for y in range(NY):
        for x in range(NX):
            rr[x][y] = [t.buffer_address() for t in ins] + [n, k * n]
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
        compile_time_args=[n, FORMS[form], OPS[op]], defines=[("EB_DUMP_PER_TILE", "1" if per_tile else "0")],
        runtime_args=[], config=ttnn.ComputeConfigDescriptor(math_fidelity=FID[fid], fp32_dest_acc_en=fp32),
    )
    pages = 8 if form == "sdpa" else 2
    out_dt = tout.dtype
    cbs = [_cb(i, ttnn.bfloat16, tile, pages, grid) for i in range(len(ins))] + [_cb(16, out_dt, tile, pages, grid)]
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    device.clear_program_cache()
    return out_bits(ttnn.generic_op(list(ins) + [tout], prog))


def _chunk(form, c, B=None):
    """(a bits, b bits, per-element b operand bits) of chunk c, CHUNK elements, as (rows, 32). B: a reduced b set (set mode)."""
    i = np.arange(CHUNK, dtype=np.int64)
    if form == "none":
        if B is None:
            a = PATS[i % 65536]
            b = PATS[(i % 65536 + c * 256 + i // 65536) % 65536]
        else:
            nb = B.size
            a = PATS[i % 65536]
            b = B[(i // 65536 + c * (CHUNK // 65536)) % nb]
        return a.reshape(-1, 32), b.reshape(-1, 32), b
    R = CHUNK // 32
    r = np.arange(R, dtype=np.int64)
    col = np.arange(32, dtype=np.int64)
    if form == "col":
        a = PATS[((r % 2048) * 32)[:, None] + col[None, :]]
        bidx = c * 256 + r // 2048
        bcol = PATS[bidx % 65536] if B is None else B[bidx % B.size]
        b = np.repeat(bcol[:, None], 32, axis=1)
        return a, b, np.repeat(bcol, 32)
    # sdpa: tile t = rows 8t..8t+7, block k = tiles 8k..8k+7; group G = c * 8192 + k: P1 row j = PATS[(G // 256) * 8 + j],
    # a of tile j', row j, column col = PATS[(G % 256) * 256 + j' * 32 + col]
    G = c * 8192 + r // 64
    jt = (r // 8) % 8
    jr = r % 8
    a = PATS[(((G % 256) * 256 + jt * 32)[:, None] + col[None, :]) % 65536]
    p1 = PATS[((G // 256) * 8 + jr) % 65536] if B is None else B[((G // 256) * 8 + jr) % B.size]
    first = (r % 64) < 8
    b = np.where(first[:, None], np.repeat(p1[:, None], 32, axis=1), np.uint16(0x3F80))
    return a, b.astype(np.uint16), np.repeat(p1, 32)


ALLF = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
# (form, ops, fidelities, dest, tile, mode): one upload of a and b per chunk serves every (op, fidelity) of the group
CFG = [("col", ("mul",), ALLF, d, (h, 32), "full") for h in (8, 4, 2, 1) for d in DEST]
CFG += [("none", ("mul",), ("HiFi4",), d, (8, 32), "full") for d in DEST]
IDS = [f"{fo}-{'_'.join(ops)}-{'_'.join(fs)}-{d}-{t[0]}x{t[1]}-{m}" for fo, ops, fs, d, t, m in CFG]


@pytest.mark.parametrize("form, ops, fids, dest, tile, mode", CFG, ids=IDS)
def test_wt(device, form, ops, fids, dest, tile, mode):
    t0 = time.time()
    fp32, out_dt = DEST[dest]
    rows = CHUNK // 32
    ntiles = rows // tile[0]
    n = ntiles // (NX * NY)
    assert n * NX * NY == ntiles and n % 8 == 0
    B = None if mode == "full" else b16_set()
    nchunks = 256 if mode == "full" else -(-B.size // 256)
    if MAX_CHUNKS:
        nchunks = min(nchunks, MAX_CHUNKS)
    tout = ttnn.from_torch(
        torch.zeros(1, 1, rows, 32), dtype=out_dt, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile(list(tile)), device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tz = _dev(np.zeros(CHUNK, dtype=np.uint16), rows, tile, device) if form == "sdpa" else None
    diffs = {
        (op, f): Diff(f"wt_{form}_{op}_{f}_{dest}_{tile[0]}x{tile[1]}_{mode} [per-face | per-tile]", op, ("PF", "PT"))
        for op in ops
        for f in fids
    }
    a, _, _ = _chunk(form, 0, B)  # a does not depend on the chunk
    ta = _dev(a, rows, tile, device)
    av = f32_of_bf16(a.reshape(-1))
    for c in range(nchunks):
        a_c, b, bsel = _chunk(form, c, B)
        assert np.array_equal(a_c, a)
        tb = _dev(b, rows, tile, device)
        ins = [ta, tb] + ([tz] if tz is not None else [])
        bv = f32_of_bf16(bsel)
        for (op, f), d in diffs.items():
            pf = _run(device, ins, tout, n, form, op, f, fp32, False, tile)
            pt = _run(device, ins, tout, n, form, op, f, fp32, True, tile)
            d.add(pf, pt, av, bv)
        ttnn.deallocate(tb)
        if c == 0:
            print(f"\nDUMP wt chunk 0 in {time.time() - t0:.1f} s", flush=True)
    for d in diffs.values():
        d.report(f"({nchunks} chunks of {CHUNK} pairs, {time.time() - t0:.1f} s for the group)")


def test_control(device):
    """Per-tile 8x32 multiply at LoFi against HiFi4 (must differ: the in-process switch reaches the kernel), and the
    per-face program at HiFi4 against itself (must not)."""
    rows, tile = CHUNK // 32, (8, 32)
    n = rows // 8 // (NX * NY)
    a, b, bsel = _chunk("none", 0)
    ta, tb = _dev(a, rows, tile, device), _dev(b, rows, tile, device)
    tout = ttnn.from_torch(torch.zeros(1, 1, rows, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([8, 32]), device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    lo = _run(device, [ta, tb], tout, n, "none", "mul", "LoFi", False, True, tile)
    hi = _run(device, [ta, tb], tout, n, "none", "mul", "HiFi4", False, True, tile)
    d = Diff("wt control none mul d16 8x32 [LoFi | HiFi4]", "mul", ("LoFi", "HiFi4"))
    d.add(lo, hi, f32_of_bf16(a.reshape(-1)), f32_of_bf16(bsel))
    d.report()
    p1 = _run(device, [ta, tb], tout, n, "none", "mul", "HiFi4", False, False, tile)
    p2 = _run(device, [ta, tb], tout, n, "none", "mul", "HiFi4", False, False, tile)
    d2 = Diff("wt control none mul d16 8x32 per-face [run 1 | run 2]", "mul", ("1", "2"))
    d2.add(p1, p2, f32_of_bf16(a.reshape(-1)), f32_of_bf16(bsel))
    d2.report()
    ac, bc, bcsel = _chunk("col", 0)
    tac, tbc = _dev(ac, rows, tile, device), _dev(bc, rows, tile, device)
    clo = _run(device, [tac, tbc], tout, n, "col", "mul", "LoFi", False, True, tile)
    chi = _run(device, [tac, tbc], tout, n, "col", "mul", "HiFi4", False, True, tile)
    d3 = Diff("wt control col mul d16 8x32 per-tile [LoFi | HiFi4]", "mul", ("LoFi", "HiFi4"))
    d3.add(clo, chi, f32_of_bf16(ac.reshape(-1)), f32_of_bf16(bcsel))
    d3.report()
    # the outputs against a host product of the device's own inputs, as a sanity check of the data path (HiFi4 bf16)
    ref = (torch.from_numpy(f32_of_bf16(a.reshape(-1))) * torch.from_numpy(f32_of_bf16(bsel))).to(torch.bfloat16)
    got = bf16_from_bits(hi)
    fin = torch.isfinite(ref.float()) & torch.isfinite(got.float())
    rel = ((ref.float() - got.float()).abs() / ref.float().abs().clamp_min(1e-30))[fin]
    print(f"\nDUMP wt control HiFi4 against host: finite {int(fin.sum())}, max rel {float(rel.max()):.3e}", flush=True)
