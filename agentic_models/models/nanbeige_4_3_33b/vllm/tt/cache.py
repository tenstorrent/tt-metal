"""Native partial-tile paged fill, preserving prefix and padding rows."""

from pathlib import Path

import ttnn


def fill_partial(cache, src, page_table, metadata, chunk_offset=0):
    assert cache.dtype in (ttnn.bfloat16, ttnn.bfloat8_b)
    if src.dtype != cache.dtype:
        src = ttnn.typecast(src, cache.dtype)
    tile_bytes = 2048 if cache.dtype == ttnn.bfloat16 else 1088
    assert cache.shape[2] == 32
    b, h, t, d = src.padded_shape
    assert metadata.shape[-1] == 3 * b and page_table.shape[0] == b
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    ct = [
        b,
        h,
        t,
        d // 32,
        page_table.buffer_aligned_page_size(),
        metadata.buffer_aligned_page_size(),
        64,
        tile_bytes,
    ]
    for tensor in (src, cache, page_table, metadata):
        ct += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    for c in range(64):
        rt[c % 8][c // 8] = [
            src.buffer_address(),
            cache.buffer_address(),
            page_table.buffer_address(),
            metadata.buffer_address(),
            c,
            chunk_offset,
        ]

    def cb(i, size, dtype):
        return ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size)],
        )

    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/fill_partial.cpp"),
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(
        kernels=[kernel],
        semaphores=[],
        cbs=[
            cb(0, 2 * tile_bytes, cache.dtype),
            cb(1, ct[4], ttnn.int32),
            cb(2, ct[5], ttnn.int32),
        ],
    )
    ttnn.generic_op([src, cache, page_table, metadata], program)


def mask_positions(positions, active):
    """Apply a device INT32 active-row vector before paged update/attention."""
    assert positions.dtype == active.dtype == ttnn.int32
    assert positions.layout == active.layout == ttnn.ROW_MAJOR_LAYOUT
    assert positions.shape == active.shape
    out = ttnn.empty(
        positions.shape,
        device=positions.device(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    n = positions.shape[-1]
    page = positions.buffer_aligned_page_size()
    ct = [n, page]
    for t in (positions, active, out):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [
        positions.buffer_address(),
        active.buffer_address(),
        out.buffer_address(),
    ]
    cb = ttnn.CBDescriptor(
        total_size=2 * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=page)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/mask_positions.cpp"),
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [positions, active, out],
        ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]),
    )
    return out


def store_rows(src, dst, mapping):
    """Copy selected BF16 rows into a preallocated 32-row terminal buffer."""
    assert src.shape == dst.shape and src.shape[-2] == 32
    assert src.dtype == dst.dtype == ttnn.bfloat16
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    page = mapping.buffer_aligned_page_size()
    ct = [src.shape[-1] // 32, page]
    for tensor in (src, dst, mapping):
        ct += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    for c in range(64):
        rt[c % 8][c // 8] = [
            src.buffer_address(),
            dst.buffer_address(),
            mapping.buffer_address(),
            c,
        ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size)],
        )
        for i, size, dtype in [(0, 4096, ttnn.bfloat16), (1, page, ttnn.int32)]
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/store_rows.cpp"),
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [src, dst, mapping],
        ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs),
    )


def remap_vector(src, dst, mapping):
    """Move only mapped persistent request rows, including in-place permutations."""
    assert src.dtype in (ttnn.int32, ttnn.uint32) and dst.dtype == src.dtype
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    ct = [
        src.shape[-1],
        dst.shape[-1],
        src.buffer_aligned_page_size(),
        dst.buffer_aligned_page_size(),
    ]
    for t in (src, dst, mapping):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [src.buffer_address(), dst.buffer_address(), mapping.buffer_address()]
    cb = ttnn.CBDescriptor(
        total_size=384,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=384)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/remap_vector.cpp"),
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [src, dst, mapping],
        ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]),
    )


def remap_counts(tensor, mapping):
    """In-place exact INT32 row permutation; each core owns complete tiles."""
    if tensor.dtype != ttnn.int32 or tensor.shape[-2] != 32 or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Penalty counts must be a tiled INT32 matrix with 32 rows")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    page = mapping.buffer_aligned_page_size()
    ct = [tensor.shape[-1] // 32, page]
    for t in (tensor, mapping):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    for c in range(64):
        rt[c % 8][c // 8] = [tensor.buffer_address(), mapping.buffer_address(), c]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.int32, page_size=size)],
        )
        for i, size in [(0, 8192), (1, page)]
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/remap_counts.cpp"),
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [tensor, mapping],
        ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs),
    )


def advance_active_seeds(seeds, active):
    """Advance RNG counters only for participating request slots."""
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    ct = [
        active.shape[-1],
        seeds.buffer_aligned_page_size(),
        active.buffer_aligned_page_size(),
    ]
    for tensor in (seeds, active):
        ct += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [seeds.buffer_address(), active.buffer_address()]
    cb = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=256)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/advance_active_seeds.cpp"),
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [seeds, active],
        ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]),
    )
