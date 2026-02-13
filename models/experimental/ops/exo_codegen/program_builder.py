# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Program builder: assembles a ProgramDescriptor from Exo-generated kernels.

Takes the generated C++ source strings and wires them together with
circular buffers, compile-time args, and runtime args to create a
complete ProgramDescriptor that can be executed via ttnn.generic_op().
"""

from __future__ import annotations

import ttnn

from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels


def build_eltwise_program(
    device,
    input_tensor,
    output_tensor,
    op: str = "identity",
    block_dim: int = 1,
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for an eltwise unary op using Exo-generated kernels.

    Args:
        device: TT device.
        input_tensor: Input tensor (on device, TILE_LAYOUT).
        output_tensor: Output tensor (on device, TILE_LAYOUT).
        op: Operation name ("identity" or "relu").
        block_dim: Compute loop block dimension (1 = flat loop).

    Returns:
        ProgramDescriptor ready for ttnn.generic_op().
    """
    # Generate kernel source code via Exo
    reader_src, compute_src, writer_src = generate_eltwise_kernels(op, block_dim)

    # Calculate tile counts and core grid
    num_tiles = input_tensor.volume() // (32 * 32)
    max_core = ttnn.CoreCoord(7, 7)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    (_, core_grid, core_group_1, core_group_2, work_per_core1, _) = ttnn.split_work_to_cores(all_cores, num_tiles)
    assert len(core_group_2.ranges()) == 0, "Exo eltwise does not support 2 core groups (tiles must divide evenly)"

    # Circular buffer setup
    cb_data_format = ttnn.bfloat16
    cb_page_size = 2 * 1024  # BFloat16 tile = 2 * 1024 bytes
    cb_total_size = 2 * cb_page_size  # Double-buffered

    in_cb = 0
    out_cb = 2

    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb,
        data_format=cb_data_format,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=cb_data_format,
        page_size=cb_page_size,
    )
    in_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in_cb_format],
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )

    # Compile-time args
    reader_ct_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [out_cb]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct_args = [work_per_core1, 1]  # [block_cnt, block_dim]

    # Runtime args per core
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    current_tile = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),
                    work_per_core1,
                    current_tile,
                ]
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    work_per_core1,
                    current_tile,
                ]
                current_tile += work_per_core1

    # Kernel descriptors using SOURCE_CODE (inline generated C++)
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=reader_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=writer_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_desc, out_cb_desc],
    )
