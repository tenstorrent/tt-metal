# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded chunk reads with no input-length/offset/count program specialization.

Offsets are the host's static prefill plan, not reads of request state. Captured
traces retain their original plan and addresses. Setup executes all three chunk
I/O descriptors before tracing; compile-only preparation does not load binaries.
"""

_HIDDEN_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t WT = 320;
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), 2048);
    const auto dst = TensorAccessor(da, get_arg_val<uint32_t>(1), 2048);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t begin = get_arg_val<uint32_t>(4);
    const uint32_t end = get_arg_val<uint32_t>(5);
    cb_reserve_back(tt::CBIndex::c_0, 1);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(scratch);
    for (uint32_t column = begin; column < end; ++column) {
        noc_async_read(src.get_noc_addr((start >> 5) * WT + column), scratch, 2048);
        noc_async_read_barrier();
        for (uint32_t row = count; row < 32; ++row) {
            const uint32_t half = ((row >> 4) * 512 + (row & 15u) * 16) / 2;
            for (uint32_t j = 0; j < 8; ++j) {
                words[half + j] = 0;
                words[half + 128 + j] = 0;
            }
        }
        noc_async_write(scratch, dst.get_noc_addr(column), 2048);
        noc_async_write_barrier();
    }
}
"""

_TOKENS_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), 64);
    const auto dst = TensorAccessor(da, get_arg_val<uint32_t>(1), 64);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    cb_reserve_back(tt::CBIndex::c_0, 1);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    for (uint32_t row = 0; row < count; ++row) {
        noc_async_read(src.get_noc_addr(start + row), scratch, 64);
        noc_async_read_barrier();
        noc_async_write(scratch, dst.get_noc_addr(row), 64);
        noc_async_write_barrier();
    }
}
"""


def _validate(source, start, count, *, hidden):
    import ttnn

    if type(start) is not int or type(count) is not int or start < 0 or start % 32 or not 1 <= count <= 32:
        raise ValueError("Expected aligned nonnegative integer start and integer count in [1,32]")
    if not ttnn.is_tensor_storage_on_device(source) or not source.is_allocated():
        raise ValueError("Chunk source must be allocated on device")
    if source.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("Chunk source must use interleaved DRAM")
    shape = tuple(source.shape)
    if hidden:
        if len(shape) != 4 or shape[:2] != (1, 1) or shape[-1] != 10240:
            raise ValueError("Expected hidden source [1,1,N,10240]")
        if source.dtype != ttnn.bfloat16 or source.layout != ttnn.TILE_LAYOUT:
            raise ValueError("Expected BF16 tiled hidden source")
        tile = source.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Expected standard 32x32 tiles")
        padded = (1, 1, (shape[-2] + 31) // 32 * 32, 10240)
    else:
        if len(shape) != 3 or shape[0] != 1 or shape[-1] != 16:
            raise ValueError("Expected token words [1,N,16]")
        if source.dtype != ttnn.uint32 or source.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Expected UINT32 row-major token words")
        padded = shape
    if tuple(source.padded_shape) != padded:
        raise ValueError("Unexpected physical padding")
    if start + count > shape[-2]:
        raise ValueError("Chunk falls outside logical source")


def _read(source, start, count, *, hidden):
    import ttnn

    _validate(source, start, count, hidden=hidden)
    output = ttnn.empty(
        (1, 1, count, 10240) if hidden else (1, count, 16),
        dtype=source.dtype,
        layout=source.layout,
        device=source.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = source.device().compute_with_storage_grid_size()
    workers = min(320, grid.x * grid.y) if hidden else 1
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    runtime = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        args = [source.buffer_address(), output.buffer_address(), start, count]
        if hidden:
            args += [320 * i // workers, 320 * (i + 1) // workers]
        runtime[core.x][core.y] = args
    compile_args = []
    for tensor in (source, output):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    page_size = 2048 if hidden else 64
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_HIDDEN_KERNEL if hidden else _TOKENS_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=compile_args,
                runtime_args=runtime,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=page_size,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=source.dtype, page_size=page_size)
                ],
            )
        ],
        semaphores=[],
    )
    ttnn.generic_op([source, output], descriptor)
    return output


def read_token_chunk(source, start, count):
    """Copy one aligned <=32-row chunk; zero every physical padding row."""
    return _read(source, start, count, hidden=True)


def read_token_words(source, start, count):
    """Copy all 16 UINT32 words per row without conversion."""
    return _read(source, start, count, hidden=False)


def prepare_chunk_io(mesh_device, *, width=10240):
    """Setup only: execute fixed I/O descriptors, synchronize, release scratch."""
    import torch

    import ttnn

    from .chunk_output import write_token_chunk

    if width != 10240:
        raise ValueError("Chunk I/O supports the target hidden width 10240")
    hidden = ttnn.from_torch(
        torch.zeros(1, 1, 32, 10240, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tokens = ttnn.from_torch(
        torch.zeros(1, 32, 16, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    chunk = read_token_chunk(hidden, 0, 32)
    words = read_token_words(tokens, 0, 32)
    write_token_chunk(chunk, hidden, 0)
    ttnn.synchronize_device(mesh_device)
    for tensor in (chunk, words, hidden, tokens):
        ttnn.deallocate(tensor)
