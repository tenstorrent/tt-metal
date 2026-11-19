# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write bounded BF16 token chunks into one preallocated logical output.

This avoids retaining every chunk and then allocating a second concatenation
copy. The source and destination are interleaved tiled [1,1,T,W], W%32==0.
Only valid logical rows are written; source/destination padding is invisible.
All addresses are device buffers. token_offset is the host's static chunk plan,
not a readback of a runtime request position.
"""
_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), 2048);
    const auto dst = TensorAccessor(da, get_arg_val<uint32_t>(1), 2048);
    const uint32_t offset = get_arg_val<uint32_t>(2);
    const uint32_t begin = get_arg_val<uint32_t>(3);
    const uint32_t end = get_arg_val<uint32_t>(4);
    const uint32_t T = get_arg_val<uint32_t>(5);
    cb_reserve_back(tt::CBIndex::c_0, 1);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    for (uint32_t column = begin; column < end; ++column) {
        noc_async_read(src.get_noc_addr(column), scratch, 2048);
        noc_async_read_barrier();
        for (uint32_t row = 0; row < T; ++row) {
            const uint32_t target = offset + row;
            const uint32_t target_tile = (target >> 5) * WT + column;
            const uint32_t sr = ((row >> 4) * 512 + (row & 15u) * 16) * 2;
            const uint32_t dr = (((target & 31u) >> 4) * 512 + (target & 15u) * 16) * 2;
            const uint64_t target_addr = dst.get_noc_addr(target_tile);
            noc_async_write(scratch + sr, target_addr + dr, 32);
            noc_async_write(scratch + sr + 512, target_addr + dr + 512, 32);
        }
        noc_async_write_barrier();
    }
}
"""


def write_token_chunk(source, destination, token_offset):
    import ttnn

    if type(token_offset) is not int:
        raise ValueError("Chunk offset must be an integer")
    for tensor in (source, destination):
        if not ttnn.is_tensor_storage_on_device(tensor) or not tensor.is_allocated():
            raise ValueError("Chunk tensors must be allocated on device")
    if (
        len(source.shape) != 4
        or len(destination.shape) != 4
        or tuple(source.shape)[:2] != (1, 1)
        or tuple(destination.shape)[:2] != (1, 1)
    ):
        raise ValueError("Expected single-request [1,1,T,W] tensors")
    rows, width = source.shape[-2], source.shape[-1]
    if not 1 <= rows <= 32 or width < 32 or width % 32 or destination.shape[-1] != width:
        raise ValueError("Expected <=32 source rows and equal tile-aligned widths")
    if not 0 <= token_offset <= destination.shape[-2] - rows:
        raise ValueError("Chunk falls outside the logical output")
    for tensor in (source, destination):
        if (
            tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Chunk outputs require BF16 tiled interleaved DRAM")
        tile = tensor.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Expected standard 32x32 tiles")
        if tuple(tensor.padded_shape) != (1, 1, (tensor.shape[-2] + 31) // 32 * 32, width):
            raise ValueError("Unexpected physical padding")
        if tensor.device() != source.device():
            raise ValueError("Source and destination must share the same mesh")
    if source.buffer_address() == destination.buffer_address():
        raise ValueError("Source and destination must not alias")
    tiles = width // 32
    grid = source.device().compute_with_storage_grid_size()
    count = min(tiles, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    runtime = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        runtime[core.x][core.y] = [
            source.buffer_address(),
            destination.buffer_address(),
            token_offset,
            tiles * i // count,
            tiles * (i + 1) // count,
            rows,
        ]
    compile_args = [tiles]
    for tensor in (source, destination):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=compile_args,
                runtime_args=runtime,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=2048,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            )
        ],
        semaphores=[],
    )
    return ttnn.generic_op([source, destination], program)
