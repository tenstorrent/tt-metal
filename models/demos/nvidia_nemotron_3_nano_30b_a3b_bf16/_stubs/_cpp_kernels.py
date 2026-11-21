import os

import ttnn

_KDIR = os.path.dirname(os.path.abspath(__file__))


def square(x):
    """cpp rung: y = x*x via ttnn.generic_op with a custom mul_tiles Metalium kernel.
    Reuses stock ttnn unary reader/writer; single custom compute kernel."""
    shape = list(x.shape)
    n_elems = 1
    for s in shape:
        n_elems *= s
    num_tiles = n_elems // (32 * 32)

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), x.dtype, ttnn.TILE_LAYOUT, x.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [x, out]

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    (_, core_grid, core_group_1, core_group_2, work_per_core1, _) = ttnn.split_work_to_cores(all_cores, num_tiles)

    cb_page = 2 * 1024
    in_cb, out_cb = 0, 16
    in_fmt = ttnn.CBFormatDescriptor(buffer_index=in_cb, data_format=ttnn.bfloat16, page_size=cb_page)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=out_cb, data_format=ttnn.bfloat16, page_size=cb_page)
    in_cb_d = ttnn.CBDescriptor(total_size=2 * cb_page, core_ranges=core_grid, format_descriptors=[in_fmt])
    out_cb_d = ttnn.CBDescriptor(total_size=2 * cb_page, core_ranges=core_grid, format_descriptors=[out_fmt])

    reader_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    writer_ct = [out_cb]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in core_group_1.ranges():
        for xx in range(cr.start.x, cr.end.x + 1):
            for yy in range(cr.start.y, cr.end.y + 1):
                reader_rt[xx][yy] = [x.buffer_address(), work_per_core1, cur]
                writer_rt[xx][yy] = [out.buffer_address(), work_per_core1, cur]
                cur += work_per_core1

    reader = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KDIR, "square_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[work_per_core1, 1],
        defines=[],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=[in_cb_d, out_cb_d])
    return ttnn.generic_op(io_tensors, prog)
