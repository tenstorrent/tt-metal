# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf: move only remapped rows of a row-major-paged state tensor (instead of gathering all 32 rows).

``map`` is an int32[32] device tensor: map[r] = source row for destination row r, or -1 (row unchanged).
Each destination row occupies ``pages_per_row`` consecutive interleaved DRAM pages in both tensors.
"""

_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t PAGE=get_compile_time_arg_val(0), PPR=get_compile_time_arg_val(1), ROWS=get_compile_time_arg_val(2);
    constexpr auto sa=TensorAccessorArgs<3>();
    constexpr auto da=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto src=TensorAccessor(sa,get_arg_val<uint32_t>(0),PAGE);
    const auto dst=TensorAccessor(da,get_arg_val<uint32_t>(1),PAGE);
    const auto mv=TensorAccessor(ma,get_arg_val<uint32_t>(2),128);
    const uint32_t core=get_arg_val<uint32_t>(3), ncores=get_arg_val<uint32_t>(4);
    Noc noc; CircularBuffer mcb(1), buf(0);
    mcb.reserve_back(1); buf.reserve_back(1);
    noc.async_read(mv,mcb,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=0});
    noc.async_read_barrier();
    volatile tt_l1_ptr int32_t* map=reinterpret_cast<volatile tt_l1_ptr int32_t*>(mcb.get_write_ptr());
    for(uint32_t r=0;r<ROWS;r++) {
        const int32_t s=map[r];
        if(s<0) continue;
        for(uint32_t p=core;p<PPR;p+=ncores) {
            noc.async_read(src,buf,PAGE,{.page_id=(uint32_t)s*PPR+p,.offset_bytes=0},{.offset_bytes=0});
            noc.async_read_barrier();
            noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(buf),dst,PAGE,{.offset_bytes=0},{.page_id=r*PPR+p,.offset_bytes=0});
            noc.async_write_barrier();
        }
    }
}
"""


def move_rows(src, dst, mapping, *, rows, page_size, pages_per_row, cores=32):
    """dst[r] = src[mapping[r]] for every r with mapping[r] >= 0 (page-granular copy)."""
    import ttnn

    grid = src.device().compute_with_storage_grid_size()
    ncores = min(cores, grid.x * grid.y, pages_per_row)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(coords):
        rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), mapping.buffer_address(), i, ncores]
    ct = [page_size, pages_per_row, rows]
    for t in (src, dst, mapping):
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=ranges,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    fmt = ttnn.bfloat16
    cbs = [
        ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=fmt, page_size=page_size)],
        ),
        ttnn.CBDescriptor(
            total_size=128,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=fmt, page_size=128)],
        ),
    ]
    ttnn.generic_op([src, mapping, dst], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))
    return dst


def state_geometry(tensor):
    """(page_size, pages_per_row) of a 32-row state tensor: TILE [32,12,128,128] or ROW_MAJOR [32,3,C]."""
    import ttnn

    shape = tuple(tensor.shape)
    elem = {ttnn.float32: 4, ttnn.bfloat16: 2}[tensor.dtype]
    if tensor.layout == ttnn.TILE_LAYOUT:
        tiles = 1
        for d in shape[1:-2]:
            tiles *= d
        tiles *= (shape[-2] // 32) * (shape[-1] // 32)
        return 32 * 32 * elem, tiles
    return shape[-1] * elem, shape[1]
