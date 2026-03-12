# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
16-combination matrix: tilize_mode × srcA_fmt × srcB_fmt × input_dtype.
All with pack_untilize_block<Wt> (standard untilize).

Kernel: ttnn/ttnn/operations/rms_norm/kernels/repro_fast_tilize_untilize.cpp
Run:
    scripts/tt-test.sh --run-all tests/ttnn/unit_tests/operations/rms_norm/test_fast_tilize_repro.py -v -s
"""

import struct
from pathlib import Path
from math import prod

import pytest
import torch
import ttnn

KERNEL_DIR = Path(__file__).resolve().parents[5] / "ttnn" / "ttnn" / "operations" / "rms_norm" / "kernels"
COMPUTE_KERNEL = str(KERNEL_DIR / "repro_fast_tilize_untilize.cpp")
READER_KERNEL = str(KERNEL_DIR / "rms_norm_reader.cpp")
WRITER_KERNEL = str(KERNEL_DIR / "rms_norm_writer.cpp")

# c2 = always bf16,  c6 = always fp32
FMT_CB = {"fp32": 6, "bf16": 2}


def _bf16_packed(v: float) -> int:
    b = struct.pack("f", v)[2:4]
    return int.from_bytes(b + b, byteorder="little")


def _f32_bits(v: float) -> int:
    return int.from_bytes(struct.pack("f", v), byteorder="little")


def run_combo(device, *, tilize, srca, srcb, data_fmt):
    dtype_map = {
        "fp32": (torch.float32, ttnn.float32),
        "bf16": (torch.bfloat16, ttnn.bfloat16),
    }
    dtype_torch, dtype_ttnn = dtype_map[data_fmt]
    shape = (1, 1, 32, 64)

    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=dtype_torch)

    x_tt = ttnn.from_torch(
        x,
        dtype=dtype_ttnn,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.zeros_like(x_tt)

    defines = [
        ("USE_FAST_TILIZE", "1" if tilize == "fast" else "0"),
        ("SRCA_INDEX", str(FMT_CB[srca])),
        ("SRCB_INDEX", str(FMT_CB[srcb])),
    ]

    pd = _build_pd(x_tt, out_tt, defines)
    ttnn.generic_op([x_tt, out_tt], pd)

    actual = ttnn.to_torch(out_tt).float()
    diff = (actual - x.float()).abs()
    md = diff.max().item()
    r16 = diff[..., 16, :].max().item() if shape[-2] > 16 else 0.0
    ok = "PASS" if md < 0.01 else "FAIL"

    print(f"  D={data_fmt:4s} til={tilize:7s} A={srca:4s} B={srcb:4s}" f"  max={md:12.3e}  row16={r16:12.3e}  [{ok}]")
    return md


def _build_pd(inp, out, defines):
    shape = inp.shape
    W = shape[-1]
    TILE_H, TILE_W = 32, 32
    Wt = W // TILE_W
    Ht_total = prod(shape[i] for i in range(len(shape) - 1)) // TILE_H

    tile_size = ttnn.tile_size(inp.dtype)
    stick_size = W * inp.element_size()
    fp32_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)

    device = inp.device()
    cg = device.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(cg.x - 1, cg.y - 1)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    _, core_grid, cg1, cg2, rpc1, rpc2 = ttnn.split_work_to_cores(all_cores, Ht_total)

    cbs = [
        # c0: RM input (input_dtype)
        ttnn.CBDescriptor(Wt * tile_size, core_grid, [ttnn.CBFormatDescriptor(0, inp.dtype, tile_size)]),
        # c1: tilized (input_dtype)
        ttnn.CBDescriptor(Wt * tile_size, core_grid, [ttnn.CBFormatDescriptor(1, inp.dtype, tile_size)]),
        # c2: always bf16 (scaler + hw_startup format source)
        ttnn.CBDescriptor(bf16_tile, core_grid, [ttnn.CBFormatDescriptor(2, ttnn.bfloat16, bf16_tile)]),
        # c5: epsilon (input_dtype, reader fills unconditionally)
        ttnn.CBDescriptor(tile_size, core_grid, [ttnn.CBFormatDescriptor(5, inp.dtype, tile_size)]),
        # c6: always fp32 (hw_startup format source, never used for data)
        ttnn.CBDescriptor(fp32_tile, core_grid, [ttnn.CBFormatDescriptor(6, ttnn.float32, fp32_tile)]),
        # c17: untilized output (input_dtype)
        ttnn.CBDescriptor(Wt * tile_size, core_grid, [ttnn.CBFormatDescriptor(17, inp.dtype, tile_size)]),
    ]

    reader_ct = [
        stick_size,
        _bf16_packed(1.0 / (Wt * 32)),
        _f32_bits(1e-6),
        1,
        0,
        Wt,
        stick_size,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
    reader_ct.append(0)

    writer_ct = [stick_size, 1, Wt, 0]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    compute_ct = [max(rpc1, rpc2 if rpc2 > 0 else 0), Wt, 1, 0]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    inp_addr, out_addr = inp.buffer_address(), out.buffer_address()
    cur = 0

    def assign(group, rpc):
        nonlocal cur
        for cr in group.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    reader_rt[x][y] = [inp_addr, cur * TILE_H, rpc, 0]
                    writer_rt[x][y] = [out_addr, cur * TILE_H, rpc]
                    compute_rt[x][y] = [rpc]
                    cur += rpc

    assign(cg1, rpc1)
    if rpc2 > 0:
        assign(cg2, rpc2)
    for y in range(cg.y):
        for x in range(cg.x):
            try:
                _ = reader_rt[x][y]
            except (KeyError, IndexError):
                reader_rt[x][y] = []
                writer_rt[x][y] = []
                compute_rt[x][y] = []

    fp32_acc = inp.dtype == ttnn.float32
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER_KERNEL,
            core_ranges=core_grid,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=WRITER_KERNEL,
            core_ranges=core_grid,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            core_ranges=core_grid,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=fp32_acc,
                math_approx_mode=False,
            ),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


# ============================================================================
# 16 combos: tilize(fast/regular) × srcA(fp32/bf16) × srcB(fp32/bf16) × data(fp32/bf16)
# ============================================================================

COMBOS = [
    (tilize, srca, srcb, data)
    for data in ["fp32", "bf16"]
    for tilize in ["fast", "regular"]
    for srca in ["fp32", "bf16"]
    for srcb in ["fp32", "bf16"]
]


@pytest.mark.parametrize(
    "tilize,srca,srcb,data_fmt",
    COMBOS,
    ids=[f"D{d}-{t[0].upper()}-A{a[0]}-B{b[0]}" for t, a, b, d in COMBOS],
)
def test_combo(device, tilize, srca, srcb, data_fmt):
    diff = run_combo(device, tilize=tilize, srca=srca, srcb=srcb, data_fmt=data_fmt)
    if diff < 0.01:
        pass
    else:
        pytest.fail(f"max_diff={diff:.3e}")
