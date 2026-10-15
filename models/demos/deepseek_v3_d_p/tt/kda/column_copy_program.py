# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A column-range tile copy between two TILE tensors, carried on spare cores of another program."""

from __future__ import annotations

import ttnn

_CB = 0

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Copy tile rows [row_begin, row_end) of `columns` tiles: source columns from source_column, destination from 0.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr auto source_args = TensorAccessorArgs<2>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const uint32_t source_addr = get_arg_val<uint32_t>(0);
    const uint32_t destination_addr = get_arg_val<uint32_t>(1);
    const uint32_t source_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t source_column = get_arg_val<uint32_t>(3);
    const uint32_t destination_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t row_begin = get_arg_val<uint32_t>(5);
    const uint32_t row_end = get_arg_val<uint32_t>(6);

    const uint32_t tile_bytes = get_tile_size(cb);
    const auto source = TensorAccessor(source_args, source_addr, tile_bytes);
    const auto destination = TensorAccessor(destination_args, destination_addr, tile_bytes);
    cb_reserve_back(cb, columns);
    const uint32_t buffer = get_write_ptr(cb);
    for (uint32_t row = row_begin; row < row_end; ++row) {
        for (uint32_t c = 0; c < columns; ++c) {
            noc_async_read_page(row * source_row_tiles + source_column + c, source, buffer + c * tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < columns; ++c) {
            noc_async_write_page(row * destination_row_tiles + c, destination, buffer + c * tile_bytes);
        }
        noc_async_write_barrier();
    }
}
"""


def copy_columns_program(
    program: ttnn.MeshProgramDescriptor,
    *,
    source: ttnn.Tensor,
    source_column: int,
    destination: ttnn.Tensor,
    columns: int,
    cores: list[ttnn.CoreCoord],
) -> ttnn.MeshProgramDescriptor:
    """Merge into `program` (on every device) a copy of `columns` columns of BF16 TILE `source` starting at
    `source_column` into the first columns of `destination`, split by tile rows over `cores`, which `program` must
    leave idle."""
    if not cores:
        raise ValueError("copy_columns_program needs at least one core")
    if source_column % 32 or columns % 32:
        raise ValueError("copy_columns_program copies whole tile columns")
    rows = source.padded_shape[-2] // 32
    cores = cores[:rows]
    per_core = -(-rows // len(cores))
    tiles = columns // 32
    core_set = ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]))
    ct = [_CB, tiles] + list(ttnn.TensorAccessorArgs(source).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(destination).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for index, core in enumerate(cores):
        begin, end = min(rows, index * per_core), min(rows, (index + 1) * per_core)
        rt[core.x][core.y] = [
            source.buffer_address(),
            destination.buffer_address(),
            source.padded_shape[-1] // 32,
            source_column // 32,
            destination.padded_shape[-1] // 32,
            begin,
            end,
        ]
    copy = ttnn.ProgramDescriptor()
    copy.cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * 2048,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=_CB, data_format=ttnn.bfloat16, page_size=2048)],
        )
    ]
    copy.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),
        )
    ]
    merged = ttnn.MeshProgramDescriptor()
    shape = tuple(source.device().shape)
    for r in range(shape[0]):
        for c in range(shape[1]):
            key = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            merged[key] = ttnn.merge_program_descriptors([program[key], copy])
    return merged
