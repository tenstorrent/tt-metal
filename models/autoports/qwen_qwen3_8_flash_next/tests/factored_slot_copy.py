"""Diagnostic-only indexed BF16 tile-row copies; no model or sampler arithmetic.

Native scatter operates through row-major output allocations and does not target
our independent fixed-address slot buffers. This narrow kernel preserves those
buffers and loads a replicated device UINT32 index payload on every replay.
"""

SOURCE = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t SCATTER = get_compile_time_arg_val(0);
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);
    constexpr auto ia = TensorAccessorArgs<2>();
    constexpr auto pa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto indices = TensorAccessor(ia, get_arg_val<uint32_t>(0), 128);
    const auto packed = TensorAccessor(pa, get_arg_val<uint32_t>(1), 2048);
    const uint32_t begin = get_arg_val<uint32_t>(2), end = get_arg_val<uint32_t>(3);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t index_buf = scratch + 2048;
    noc_async_read(indices.get_noc_addr(0), index_buf, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_buf);
    for (uint32_t col = begin; col < end; ++col) {
        if constexpr (SCATTER) {
            noc_async_read(packed.get_noc_addr(col), scratch, 2048);
            noc_async_read_barrier();
        }
        for (uint32_t row = 0; row < ROWS; ++row) {
            const uint32_t slot = slots[row];
            // Host membership validation guarantees unique values in [0,31].
            const auto single = TensorAccessor(ra, get_arg_val<uint32_t>(4 + slot), 2048);
            const uint32_t row_offset = ((row >> 4) * 512 + (row & 15u) * 16) * 2;
            if constexpr (SCATTER) {
                const uint64_t dst = single.get_noc_addr(col);
                noc_async_write(scratch + row_offset, dst, 32);
                noc_async_write(scratch + row_offset + 512, dst + 512, 32);
            } else {
                noc_async_read(single.get_noc_addr(col), scratch, 2048);
                noc_async_read_barrier();
                const uint64_t dst = packed.get_noc_addr(col);
                noc_async_write(scratch, dst + row_offset, 32);
                noc_async_write(scratch + 512, dst + row_offset + 512, 32);
                // Scratch cannot be overwritten by the next read until writes finish.
                noc_async_write_barrier();
            }
        }
        noc_async_write_barrier();
    }
}
"""


def validate_slots(slots):
    slots = tuple(slots)
    if not 1 <= len(slots) <= 32 or any(type(s) is not int or not 0 <= s < 32 for s in slots):
        raise ValueError("Expected 1..32 integer slots in [0,31]")
    if slots != tuple(sorted(set(slots))):
        raise ValueError("Expected unique slots in ascending model arithmetic order")
    return slots


def indexed_copy(indices, packed, rows, *, scatter=False):
    """Copy valid logical rows; padding and inactive slot storage are untouched."""
    import ttnn as tt

    if len(rows) != 32 or not 1 <= packed.shape[-2] <= 32:
        raise ValueError("Expected 32 persistent row buffers and <=32 packed rows")
    if tuple(indices.shape) != (32,) or indices.dtype != tt.uint32 or indices.layout != tt.ROW_MAJOR_LAYOUT:
        raise ValueError("Indices require UINT32 row-major [32]")
    if indices.memory_config() != tt.DRAM_MEMORY_CONFIG:
        raise ValueError("Indices require interleaved DRAM")
    width = packed.shape[-1]
    if width % 32 or tuple(packed.shape)[:2] != (1, 1):
        raise ValueError("Expected [1,1,N,W] tile-aligned features")
    for tensor in [packed, *rows]:
        if (
            tensor.dtype != tt.bfloat16
            or tensor.layout != tt.TILE_LAYOUT
            or tensor.memory_config() != tt.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Copy tensors require BF16 tiled interleaved DRAM")
        if tensor.device() != packed.device() or not tensor.is_allocated():
            raise ValueError("Copy tensors must be allocated on the same mesh")
        tile = tensor.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Expected standard 32x32 tile faces")
    if any(tuple(row.shape) != (1, 1, 1, width) for row in rows):
        raise ValueError("Slot rows require [1,1,1,W]")
    if len({tensor.buffer_address() for tensor in [packed, *rows]}) != 33:
        raise ValueError("Copy storage must not alias")
    grid = packed.device().compute_with_storage_grid_size()
    columns = width // 32
    count = min(columns, grid.x * grid.y)
    cores = [tt.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    ranges = tt.CoreRangeSet([tt.CoreRange(core, core) for core in cores])
    runtime = tt.RuntimeArgs()
    for i, core in enumerate(cores):
        runtime[core.x][core.y] = [
            indices.buffer_address(),
            packed.buffer_address(),
            columns * i // count,
            columns * (i + 1) // count,
            *[row.buffer_address() for row in rows],
        ]
    args = [int(scatter), packed.shape[-2]]
    for tensor in (indices, packed, rows[0]):
        args.extend(tt.TensorAccessorArgs(tensor).get_compile_time_args())
    program = tt.ProgramDescriptor(
        kernels=[
            tt.KernelDescriptor(
                kernel_source=SOURCE,
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=args,
                runtime_args=runtime,
                config=tt.WriterConfigDescriptor(),
            )
        ],
        cbs=[
            tt.CBDescriptor(
                total_size=2176,
                core_ranges=ranges,
                format_descriptors=[tt.CBFormatDescriptor(buffer_index=0, data_format=tt.uint32, page_size=2176)],
            )
        ],
        semaphores=[],
    )
    return tt.generic_op([indices, packed, *rows], program)


def cpu_check():
    """Exhaustively check BF16 face offsets and inactive-byte preservation."""
    import numpy as np

    rng = np.random.default_rng(123)
    source = rng.integers(0, 65536, size=(32, 1024), dtype=np.uint16)
    for width in range(1, 33):
        slots = validate_slots(tuple(sorted(rng.choice(32, width, replace=False).tolist())))
        packed = np.full(1024, 0xCAFE, dtype=np.uint16)
        output = np.full_like(source, 0xDEAD)
        for row, slot in enumerate(slots):
            offset = (row // 16) * 512 + (row % 16) * 16
            for face in (0, 256):
                packed[offset + face : offset + face + 16] = source[slot, face : face + 16]
                output[slot, face : face + 16] = packed[offset + face : offset + face + 16]
        for slot in range(32):
            expected = np.full(1024, 0xDEAD, dtype=np.uint16)
            if slot in slots:
                expected[:16] = source[slot, :16]
                expected[256:272] = source[slot, 256:272]
            assert np.array_equal(expected, output[slot])
    for bad in ((), (1, 1), (2, 1), (32,), (True,)):
        try:
            validate_slots(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    print("CPU indexed tile-face/slot validation passed; no device claim")
