# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Template Op - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.
"""

from pathlib import Path
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the template operation.

    Args:
        input_tensor: Input tensor (on device)
        output_tensor: Pre-allocated output tensor (on device)

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    # Prefer buffer query methods — they work for any layout (tiled or row-major)
    # and account for padding/alignment automatically.
    #
    # Buffer query APIs (layout-agnostic, preferred):
    #   tensor.buffer_page_size()          # page size in bytes (tile size or stick size)
    #   tensor.buffer_aligned_page_size()  # page size rounded to DRAM/L1 alignment
    #   tensor.buffer_num_pages()          # total pages (tiles or sticks)
    #
    # Standalone utilities (for intermediate CBs with no tensor):
    #   ttnn.tile_size(dtype)              # tile size in bytes for a 32x32 tile
    #   ttnn.round_up(val, mult)           # round up to nearest multiple
    #   ttnn.div_up(a, b)                  # ceiling division
    #   ttnn.get_dram_alignment()          # DRAM alignment in bytes (e.g. 32)
    #   ttnn.get_l1_alignment()            # L1 alignment in bytes (e.g. 16)
    #
    # Layout-specific APIs (still valid, use when you need explicit control):
    #   TILE_LAYOUT:      tensor.tile.get_tile_size(dtype)  — tile size in bytes
    #   ROW_MAJOR_LAYOUT: tensor.padded_shape[-1] * tensor.element_size()  — stick size
    #
    # Other tensor metadata:
    #   tensor.dtype                       # DataType enum (bfloat16, float32, ...)
    #   tensor.layout                      # Layout enum (TILE_LAYOUT, ROW_MAJOR_LAYOUT)
    #   tensor.element_size()              # bytes per element (2 for bf16, 4 for f32)
    #   tensor.tile.tile_shape             # (height, width), e.g. (32, 32)
    #   tensor.padded_shape                # shape with padding
    #   tensor.volume()                    # total element count

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Example: use single core for simplicity
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices: 0-7 for input, 16-23 for output, 24-31 for intermediate
    cb_in = 0
    cb_out = 16

    cb_in_descriptor = ttnn.CBDescriptor(
        total_size=2 * input_page_size,  # double buffer
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_in,
                data_format=input_tensor.dtype,
                page_size=input_page_size,
            )
        ],
    )

    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=2 * output_page_size,  # double buffer
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_out,
                data_format=output_tensor.dtype,
                page_size=output_page_size,
            )
        ],
    )

    # ========== 4. KERNEL DESCRIPTORS ==========
    # Reader kernel
    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_pages,
        0,  # start_page_id
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "template_op_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer kernel
    writer_ct_args = [cb_out]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_pages,
        0,  # start_page_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "template_op_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel
    compute_ct_args = [num_pages]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "template_op_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in_descriptor, cb_out_descriptor],
    )
