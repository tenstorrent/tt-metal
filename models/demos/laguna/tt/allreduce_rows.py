# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 decode all-reduce + residual add on row 0 only (Laguna; kernels/ar_*.cpp).

A batch-1 decode partial is a [32, H] bf16 TILE tensor whose only real row is row 0. The stock path all-gathers the
whole 32-row tiles (192 KB per chip at H = 3072, ~20 us on the p150x4 ring) and sums them. Here row 0 is packed into
one [1, H] row-major row (6 KB; its all_gather takes ~9 us), gathered, and the D rows plus the residual's row 0 are
summed in fp32 on 32 cores and written as row 0 of zero-padded output tiles: out = residual + sum of the partials."""

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def _cb(grid, i, fmt, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
    )


def _grid(device, H):
    tiles = H // 32
    per = 3 if tiles % 3 == 0 and tiles // 3 <= 64 else 1
    while (tiles // per) > 64 or tiles % per:
        per += 1
    cores = tiles // per
    gs = device.compute_with_storage_grid_size()
    return per, cores, gs, ttnn.num_cores_to_corerangeset(cores, gs, True)


def pack_row0(x, rows=1):
    """x: [.., 32(padded), H] bf16 TILE (interleaved or sharded). Returns [1, 1, rows, H] bf16 row-major, L1: rows
    0..rows-1 (a batch-``rows`` decode partial; rows <= 10 at H = 3072)."""
    device = x.device()
    H = x.shape[-1]
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and H % 32 == 0, (x.dtype, x.layout, H)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device,
                                         ttnn.L1_MEMORY_CONFIG)
    per, cores, gs, grid = _grid(device, H)
    acc = list(ttnn.TensorAccessorArgs(x).get_compile_time_args()) + list(
        ttnn.TensorAccessorArgs(out).get_compile_time_args()
    )
    k = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ar_pack_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[per, gs.x, rows] + acc,
        common_runtime_args=[x.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    assert rows * per * 32 <= 1024, (rows, per)  # the summed rows of one core fit one tile-sized slot
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(grid, 0, ttnn.bfloat16, per * 64 * rows, 1), _cb(grid, 1, ttnn.bfloat16, per * 128 * rows, 1)]))
    return out


def sum_rows_add(g, residual, memory_config=None):
    """g: [1, D, R, H] bf16 row-major (the gathered packed rows) or a list of D [1, 1, R, H] row-major tensors (the
    all_broadcast outputs); residual: [.., 32(padded), H] bf16 TILE. Returns residual's shape/layout in memory_config
    (default residual's): rows 0..R-1 = the residual's rows + the sum of the D partials' rows (fp32 adds, bf16
    result), rows R..31 zero."""
    split = isinstance(g, (list, tuple))
    rows = list(g) if split else [g]
    device = rows[0].device()
    D, H = (len(rows) if split else g.shape[1]), rows[0].shape[-1]
    R = rows[0].shape[-2]
    per, cores, gs, grid = _grid(device, H)
    mem = memory_config or residual.memory_config()
    out = ttnn.allocate_tensor_on_device(residual.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ar_sum_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[per, gs.x, D, int(split), R]
        + list(ttnn.TensorAccessorArgs(rows[0]).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(residual).get_compile_time_args()),
        common_runtime_args=[rows[0].buffer_address(), residual.buffer_address()]
        + ([t.buffer_address() for t in rows] if split else []),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ar_sum_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[per, gs.x, R] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = True
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ar_sum_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[D],
        config=cfg,
    )
    cbs = [
        _cb(grid, 0, ttnn.bfloat16, 2048, D + 1),
        _cb(grid, 1, ttnn.bfloat16, per * 128 * R, 1),
        _cb(grid, 16, ttnn.bfloat16, 2048, 1),
        _cb(grid, 17, ttnn.bfloat16, 2048, per),
    ]
    ttnn.generic_op(
        rows + [residual, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    )
    return out
