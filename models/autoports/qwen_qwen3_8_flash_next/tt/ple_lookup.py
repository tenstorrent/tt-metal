"""Device-only lookup of 160-wide rows from uniformly sharded BFP4 tables.

The table remains compressed in DRAM; a dataflow kernel decodes only requested
rows into BF16. This is needed because native embedding requires BF16 weights.
No table values, indices, or output data cross the host at runtime.
"""

from pathlib import Path

import ttnn


def lookup_rows(tables, indices, *, output=None):
    """Look up ``indices [N,16]`` in equal-sized ``tables [R,160]`` shards.

    Global indices map to (shard=index//R, row=index%R). Indices must be valid
    UINT32 row-major device values. Returns BF16 row-major [N,2560].
    ``output`` can be preallocated before capture for a stable output binding.
    """
    rows = tables[0].shape[-2]
    if len(tables) > 128:
        raise ValueError("At most 128 checkpoint shards are supported")
    for table in tables:
        if tuple(table.shape) != (rows, 160) or table.dtype != ttnn.bfloat4_b or table.layout != ttnn.TILE_LAYOUT:
            raise ValueError("PLE tables must be equally sized BFP4 TILE [rows,160]")
        if table.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("PLE tables must use interleaved DRAM")
    if indices.dtype != ttnn.uint32 or indices.layout != ttnn.ROW_MAJOR_LAYOUT or indices.shape[-1] != 16:
        raise ValueError("PLE indices must be UINT32 ROW_MAJOR [N,16]")
    n = indices.shape[0]
    mesh = indices.device()
    if output is None:
        output = ttnn.allocate_tensor_on_device(
            (n, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    grid = mesh.compute_with_storage_grid_size()
    _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, n, True)
    work = []
    start = 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        count = work1 if i < group1.num_cores() else work2
        work.append(
            (
                (core.x, core.y),
                [
                    indices.buffer_address(),
                    output.buffer_address(),
                    start,
                    count,
                    *[t.buffer_address() for t in tables],
                ],
            )
        )
        start += count
    args = [
        rows,
        len(tables),
        *ttnn.TensorAccessorArgs(indices).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(tables[0]).get_compile_time_args(),
    ]
    cb = ttnn.CBDescriptor(
        total_size=8192,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=8192)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "ple_lookup.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=work,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([*tables, indices, output], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return output
