# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer: fast_tilize → pack_untilize corrupts face 2/3 for fp32 + Wt>1.

The compute kernel does ONLY tilize → untilize (identity, no math).
Toggle USE_FAST_TILIZE in the kernel to switch between fast (buggy) and regular (correct).

Kernel: ttnn/ttnn/operations/rms_norm/kernels/repro_fast_tilize_untilize.cpp

Run:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/rms_norm/test_fast_tilize_repro.py -v -s
"""

import struct
from pathlib import Path
from math import prod

import torch
import ttnn

# Paths to kernels
KERNEL_DIR = Path(__file__).resolve().parents[5] / "ttnn" / "ttnn" / "operations" / "rms_norm" / "kernels"
COMPUTE_KERNEL = str(KERNEL_DIR / "repro_fast_tilize_untilize.cpp")
READER_KERNEL = str(KERNEL_DIR / "rms_norm_reader.cpp")
WRITER_KERNEL = str(KERNEL_DIR / "rms_norm_writer.cpp")


def _float_to_bfloat16_packed(value: float) -> int:
    b = struct.pack("f", value)
    bf16 = b[2:4]
    return int.from_bytes(bf16 + bf16, byteorder="little")


def _float_to_uint32(value: float) -> int:
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def run_tilize_untilize_roundtrip(device, shape, dtype_torch, dtype_ttnn):
    """Send RM data through tilize→untilize kernel, compare to input."""
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

    pd = _build_program_descriptor(x_tt, out_tt)
    ttnn.generic_op([x_tt, out_tt], pd)

    actual = ttnn.to_torch(out_tt).float()
    expected = x.float()
    diff = (actual - expected).abs()

    # Per-face-boundary report
    print(f"\n  shape={shape} dtype={dtype_ttnn}")
    print(f"  max_diff = {diff.max().item():.6e}")
    for r in [0, 15, 16, 31]:
        if r < shape[-2]:
            print(f"  row {r:2d} max: {diff[..., r, :].max().item():.6e}")

    return diff.max().item()


def _build_program_descriptor(inp, out):
    """Minimal ProgramDescriptor: reader → tilize→untilize compute → writer."""
    shape = inp.shape
    W = shape[-1]
    TILE_H, TILE_W = 32, 32
    Wt = W // TILE_W
    Ht_total = prod(shape[i] for i in range(len(shape) - 1)) // TILE_H

    tile_size = ttnn.tile_size(inp.dtype)
    elem_size = inp.element_size()
    stick_size = W * elem_size

    device = inp.device()
    cg = device.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(cg.x - 1, cg.y - 1)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    num_cores, core_grid, cg1, cg2, rpc1, rpc2 = ttnn.split_work_to_cores(all_cores, Ht_total)

    # ---- Circular Buffers (only what's needed) ----
    scaler_tile = ttnn.tile_size(ttnn.bfloat16)
    cbs = [
        # c0: RM input sticks (Wt tile-sized pages)
        ttnn.CBDescriptor(Wt * tile_size, core_grid, [ttnn.CBFormatDescriptor(0, inp.dtype, tile_size)]),
        # c1: tilized tiles
        ttnn.CBDescriptor(Wt * tile_size, core_grid, [ttnn.CBFormatDescriptor(1, inp.dtype, tile_size)]),
        # c2: scaler (reader generates unconditionally)
        ttnn.CBDescriptor(scaler_tile, core_grid, [ttnn.CBFormatDescriptor(2, ttnn.bfloat16, scaler_tile)]),
        # c5: epsilon (reader generates unconditionally)
        ttnn.CBDescriptor(tile_size, core_grid, [ttnn.CBFormatDescriptor(5, inp.dtype, tile_size)]),
        # c17: untilized output
        ttnn.CBDescriptor(Wt * tile_size, core_grid, [ttnn.CBFormatDescriptor(17, inp.dtype, tile_size)]),
    ]

    # ---- Reader compile-time args ----
    reader_ct = [
        stick_size,  # 0: stick_or_tile_size
        _float_to_bfloat16_packed(1.0 / (Wt * 32)),  # 1: scaler
        _float_to_uint32(1e-6),  # 2: eps
        1,  # 3: input_is_rm
        0,  # 4: has_gamma
        Wt,  # 5: Wt
        stick_size,  # 6: gamma_stick_size (unused)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
    reader_ct.append(0)  # gamma accessor placeholder

    # ---- Writer compile-time args ----
    writer_ct = [stick_size, 1, Wt, 0]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    # ---- Compute compile-time args ----
    compute_ct = [
        max(rpc1, rpc2 if rpc2 > 0 else 0),  # 0: Ht_max
        Wt,  # 1: Wt
        1,  # 2: input_is_rm
        0,  # 3: has_gamma (unused by repro kernel)
    ]

    # ---- Runtime args (per core) ----
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inp_addr = inp.buffer_address()
    out_addr = out.buffer_address()
    cur = 0

    def assign(group, rpc):
        nonlocal cur
        for cr in group.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    s = cur * TILE_H
                    reader_rt[x][y] = [inp_addr, s, rpc, 0]
                    writer_rt[x][y] = [out_addr, s, rpc]
                    compute_rt[x][y] = [rpc]
                    cur += rpc

    assign(cg1, rpc1)
    if rpc2 > 0:
        assign(cg2, rpc2)

    # idle cores
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
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=fp32_acc,
                math_approx_mode=False,
            ),
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


# ============================================================================
# Tests
# ============================================================================


def test_fp32_wt2(device):
    """fp32, Wt=2 — FAILS with USE_FAST_TILIZE=1, PASSES with =0."""
    diff = run_tilize_untilize_roundtrip(device, (1, 1, 32, 64), torch.float32, ttnn.float32)
    assert diff < 0.01, f"Round-trip failed: max_diff={diff}"


def test_fp32_wt1(device):
    """fp32, Wt=1 — always passes (sanity check)."""
    diff = run_tilize_untilize_roundtrip(device, (1, 1, 32, 32), torch.float32, ttnn.float32)
    assert diff < 0.01, f"Round-trip failed: max_diff={diff}"


def test_bf16_wt2(device):
    """bf16, Wt=2 — always passes (not affected by the bug)."""
    diff = run_tilize_untilize_roundtrip(device, (1, 1, 32, 64), torch.bfloat16, ttnn.bfloat16)
    assert diff < 0.01, f"Round-trip failed: max_diff={diff}"
