"""Mirror the existing BF8 fast-reduce descriptor while selecting the dataflow NoCs."""

import ttnn

BASE = "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/"


def partition(grid_x, grid_y):
    assert grid_x > 0 and grid_y > 0
    cores = min(grid_x * grid_y, 256)
    return [(i % grid_x, i // grid_x, i, len(range(i, 256, cores))) for i in range(cores)]


def core_set(work):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _, _ in work])


def program(source, output, mesh, read_noc):
    assert read_noc in (0, 1)
    grid = mesh.compute_with_storage_grid_size()
    work = partition(grid.x, grid.y)
    cores = core_set(work)
    count = len(work)
    reader_args, writer_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for x, y, first, tiles in work:
        reader_args[x][y] = [source.buffer_address(), 8, tiles * count, first, 0, 2048, 256]
        writer_args[x][y] = [output.buffer_address(), tiles * count, first]
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )
        for index, pages, size, dtype in [
            (0, 16, 1088, ttnn.bfloat8_b),
            (1, 1, 2048, ttnn.bfloat16),
            (24, 1, 1088, ttnn.bfloat8_b),
            (16, 2, 1088, ttnn.bfloat8_b),
        ]
    ]
    nocs = [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1]
    reader = ttnn.KernelDescriptor(
        kernel_source=BASE + "reader_reduce_nc.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[8, 1, count] + list(ttnn.TensorAccessorArgs(source).get_compile_time_args()),
        runtime_args=reader_args,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=nocs[read_noc]),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=BASE + "writer_reduce_nc.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[1, count] + list(ttnn.TensorAccessorArgs(output).get_compile_time_args()),
        runtime_args=writer_args,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=nocs[1 - read_noc]),
    )
    compute = [
        ttnn.KernelDescriptor(
            kernel_source=BASE + "reduce_nc.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set([row for row in work if row[3] == tiles]),
            compile_time_args=[tiles, 8, 8],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
                math_approx_mode=True,
            ),
        )
        for tiles in sorted({row[3] for row in work}, reverse=True)
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, *compute], cbs=cbs, semaphores=[])
