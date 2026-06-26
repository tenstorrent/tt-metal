# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# lreg_layout_op: experimental op that fills a single uint32 tile with
# SFPU-generated lane/iteration IDs to visualize the LReg-iteration layout
# inside a tile.
#
# Stage 1 (this scaffold) just copies the input tile through DST. The
# placeholder compute kernel is replaced in Stage 2 by a custom SFPU LLK.

from __future__ import annotations

from math import prod

import ttnn

_TILE_ELEMENTS = 32 * 32
_UINT32_BYTES = 4
_TILE_SIZE = _UINT32_BYTES * _TILE_ELEMENTS  # 4096


def lreg_layout_op(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Run the lreg_layout custom op on a uint32, tile-layout, DRAM-interleaved tensor.

    Stage 1: copy through. Stage 2 will overwrite with SFPU IDs.
    """
    assert input_tensor.dtype == ttnn.uint32, f"input must be uint32, got {input_tensor.dtype}"
    assert input_tensor.layout == ttnn.TILE_LAYOUT, "input must be in TILE_LAYOUT"
    assert input_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "input must be DRAM interleaved"

    device = input_tensor.device()

    output = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    total_tiles = prod(input_tensor.padded_shape) // _TILE_ELEMENTS

    # Single core for this experiment (we only have 1 tile).
    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ---- Circular buffers (double-buffered) ---------------------------------
    def _cb_desc(cb_index: int) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=ttnn.uint32,
            page_size=_TILE_SIZE,
        )
        return ttnn.CBDescriptor(
            total_size=2 * _TILE_SIZE,
            core_ranges=core_set,
            format_descriptors=[fmt],
        )

    cb_in_desc = _cb_desc(0)   # c_0: input
    cb_out_desc = _cb_desc(2)  # c_2: output

    # ---- Reuse the standard unary reader/writer kernels ---------------------
    # Reader: reader_unary_interleaved_start_id.cpp
    #   ct_args: TensorAccessorArgs<0> for input
    #   rt_args: [src_addr, num_pages, start_id]
    #   hardcodes input CB = 0 (matches our compute kernel's c_in).
    # Writer: writer_unary_interleaved_start_id.cpp
    #   ct_args: [cb_id_out=2] then TensorAccessorArgs<1> for output
    #   rt_args: [dst_addr, num_pages, start_id]
    reader_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
    writer_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
    compute_path = "custom_op/lreg_layout/operation/kernels/compute/compute_lreg_layout.cpp"

    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct_args = [2] + ttnn.TensorAccessorArgs(output).get_compile_time_args()
    compute_ct_args = []

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    reader_rt_args[core.x][core.y] = [input_tensor.buffer_address(), total_tiles, 0]
    writer_rt_args[core.x][core.y] = [output.buffer_address(), total_tiles, 0]
    compute_rt_args[core.x][core.y] = [total_tiles]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=reader_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=reader_ct_args,
        defines=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    reader_kernel.runtime_args = reader_rt_args

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=writer_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=writer_ct_args,
        defines=[],
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_kernel.runtime_args = writer_rt_args

    # uint32 in DST requires fp32_dest_acc_en (32-bit slots).
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=compute_ct_args,
        defines=[],
        config=compute_cfg,
    )
    compute_kernel.runtime_args = compute_rt_args

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in_desc, cb_out_desc],
    )

    return ttnn.generic_op([input_tensor, output], program)
