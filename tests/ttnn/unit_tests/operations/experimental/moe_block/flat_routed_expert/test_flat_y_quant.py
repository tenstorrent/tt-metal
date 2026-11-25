# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flat expert's y quantize (flat_routed_expert/device/kernels/y_quant.hpp) on one core, bit exact against the torch
reference (moe_block/quantized_row_golden.py): fp8 (e4m3 x e8m0) and int8 (sign-magnitude, read as Bfp8_b).

The test kernel (kernels/y_quant_compute.cpp) runs it as the down compute does for GLM: fp32 DEST in half sync, each row
tile (6 tiles: a down core's 192 columns) in passes of 3 tiles, quantized on the PACK thread, pack-untilized into a
UInt8 CB; each row's 12 exponent bytes (one a 16 values) into its 16-byte cell."""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_spec = importlib.util.spec_from_file_location(
    "quantized_row_golden", Path(__file__).resolve().parent.parent / "quantized_row_golden.py"
)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)

TILE = 32
COLUMNS = 6  # tiles a row tile (GLM's down core)
PASS = 3  # tiles a pass
_RowFormat = ttnn._ttnn.operations.experimental.moe_block.RoutedRowFormat
FORMATS = {"fp8": _RowFormat.Fp8E4m3.value, "int8": _RowFormat.Int8.value}  # quantized_row::Format (kernel arg)
KERNELS = "tests/ttnn/unit_tests/operations/experimental/moe_block/flat_routed_expert/kernels"


def make_input(row_tiles, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = row_tiles * TILE
    y = torch.randn(rows, COLUMNS * TILE, generator=g)
    # per (row, 32 columns) magnitudes over a wide range: 2^-30 .. 2^20
    mag = torch.pow(2.0, torch.randint(-30, 21, (rows, COLUMNS), generator=g).float())
    y = (y.reshape(rows, -1, TILE) * mag[..., None]).reshape(rows, -1)
    # special rows: all zero, one outlier, amax exactly at the format's limit x 2^k and just above / below, tiny
    # (under the fp8 1e-4 floor), every rounding tie of e4m3 and of int8, values around 2^-6 of the block max
    y[0] = 0
    y[1] = torch.randn(COLUMNS * TILE, generator=g) * 1e-3
    y[1, 5] = 1e4
    y[2, :96] = torch.linspace(-448, 448, 96)
    y[2, 96:] = torch.linspace(-127 * 4, 127 * 2, 96)
    y[3] = y[2] * (1 + 2**-20)
    y[4] = y[2] * (1 - 2**-20)
    y[5] = torch.randn(COLUMNS * TILE, generator=g) * 1e-7
    y[6, :96] = (torch.arange(96).float() + 0.5) * (448 / 128)
    y[6, 96:] = torch.arange(96).float() - 47.5  # int8 ties (x.5)
    y[7, :] = torch.linspace(0.9, 1.0, COLUMNS * TILE) * 2**-6 * (torch.arange(COLUMNS * TILE) % 2 * 2 - 1)
    y[7, 0] = 448.0
    y[7, 32] = 448.0
    y[7, 64] = 448.0
    return y


def run(device, y, fmt, block_tiles):
    rows = y.shape[0]
    row_tiles = rows // TILE
    stride = COLUMNS + (PASS if PASS % 2 else 0)  # (an odd pass width into 8 bits writes a tile past the pass)
    t_in = ttnn.from_torch(
        y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def rm_u8(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    t_y = rm_u8((rows, COLUMNS * TILE))
    t_c = rm_u8((row_tiles, 32 * 16))
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(idx, fmt_, page, pages):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt_, page_size=page)],
        )

    cbs = [cb(0, ttnn.float32, 4096, 2 * COLUMNS), cb(16, ttnn.uint8, 1024, 2 * stride), cb(17, ttnn.uint8, 512, 2)]
    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, dst_full_sync_en=False)
    # (the fp32 input straight into DEST: through SrcA it would lose mantissa bits)
    compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] + [ttnn.UnpackToDestMode.Default] * 63
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [t_in.buffer_address(), row_tiles * COLUMNS, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [t_y.buffer_address(), t_c.buffer_address()]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(t_in).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/y_quant_writer.cpp",
            core_ranges=core,
            compile_time_args=[row_tiles, stride, COLUMNS * TILE, stride * TILE]
            + ttnn.TensorAccessorArgs(t_y).get_compile_time_args()
            + ttnn.TensorAccessorArgs(t_c).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/y_quant_compute.cpp",
            core_ranges=core,
            compile_time_args=[FORMATS[fmt], row_tiles, COLUMNS, PASS, block_tiles, stride],
            runtime_args=[],
            config=compute_config,
        ),
    ]
    ttnn.generic_op([t_in, t_y, t_c], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return ttnn.to_torch(t_y).to(torch.uint8), ttnn.to_torch(t_c).to(torch.uint8).reshape(rows, 16)


@pytest.mark.parametrize("block_tiles", [3, 1], ids=["1x96", "1x32"])
@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("row_tiles", [1, 4])
def test_y_quant(device, row_tiles, fmt, block_tiles):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    y = make_input(row_tiles)
    want_codes, want_e16, _ = golden.quantize(y, fmt, block_tiles * TILE)
    codes, cells = run(device, y, fmt, block_tiles)
    want_cells = golden.cells(want_e16, 1)
    bad_c = (cells != want_cells).nonzero()
    bad = (codes != want_codes).nonzero()
    for r, c in bad[:8].tolist():
        logger.warning(f"  ({r},{c}) y {y[r, c].item():.6g} got 0x{codes[r, c]:02x} want 0x{want_codes[r, c]:02x}")
    for r, c in bad_c[:4].tolist():
        logger.warning(f"  cell ({r},{c}) got {cells[r, c]} want {want_cells[r, c]}")
    logger.info(
        f"{fmt} 1x{block_tiles * TILE} row_tiles {row_tiles}: {len(bad)} of {codes.numel()} bytes differ, "
        f"{len(bad_c)} of {cells.numel()} cell bytes differ"
    )
    assert len(bad) == 0 and len(bad_c) == 0
