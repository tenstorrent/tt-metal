# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# mul_relu_overlap_block_op: experimental fused y = relu(a * b) via ttnn.generic_op.
#
# Block-based sequential baseline (no FPU/SFPU overlap yet): mul_tiles (FPU/MATH)
# -> relu_tile (SFPU/MATH), all on TRISC1, processing BS tiles per acquire/commit
# window. A future iteration will introduce overlap.
#
# Block size is dtype-driven and fixed by the operation (compile-time on all
# three kernels). DST half-sync capacity sets the cap:
#   bfloat16 -> BS=8, fp32_dest_acc_en=False (16-bit DST holds 8 tiles)
#   float32  -> BS=4, fp32_dest_acc_en=True  (32-bit DST holds 4 tiles)
#
# Reader and writer push/pop exactly BS tiles per CB transaction. For the
# tail block on a core, the reader fills only the real tiles and leaves the
# unused slots uninitialized; the writer drops those slots. So the compute
# kernel always sees full BS-tile blocks.
#
# Supported scope:
#   - dtype:   bfloat16 or float32 (both inputs must match)
#   - memory:  interleaved DRAM, tile layout
#   - shapes:  same shape for a, b, output (no broadcast, no sharding)

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import ttnn


_TILE_H = 32
_TILE_W = 32
_TILE_ELEMENTS = _TILE_H * _TILE_W


@dataclass
class MulReluOverlapBlockConfig:
    """Placeholder for future overlap toggles. Empty for now."""

    pass


def _dtype_params(dtype):
    """Return (block_size, fp32_dest_acc_en, bytes_per_elem) for the supported dtypes."""
    if dtype == ttnn.bfloat16:
        return 8, False, 2
    if dtype == ttnn.float32:
        return 4, True, 4
    raise AssertionError(f"Unsupported dtype: {dtype}")


def mul_relu_overlap_block_op(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    config: MulReluOverlapBlockConfig = MulReluOverlapBlockConfig(),
) -> ttnn.Tensor:
    """y = relu(a * b), elementwise, bfloat16 or float32, DRAM-interleaved tile layout."""
    # --- Validation ---------------------------------------------------------
    dtype = input_a.dtype
    assert dtype in (ttnn.bfloat16, ttnn.float32), f"Only bfloat16/float32 supported, got {dtype}"
    assert input_b.dtype == dtype, f"dtype mismatch a vs b: {dtype} vs {input_b.dtype}"
    assert input_a.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_b.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_a.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input A must be in DRAM (interleaved)"
    assert input_b.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input B must be in DRAM (interleaved)"
    assert list(input_a.shape) == list(input_b.shape), f"Shape mismatch a vs b: {input_a.shape} vs {input_b.shape}"

    device = input_a.device()
    block_size, fp32_dest_acc_en, bytes_per_elem = _dtype_params(dtype)
    tile_size_bytes = _TILE_ELEMENTS * bytes_per_elem

    # --- Output tensor ------------------------------------------------------
    output = ttnn.allocate_tensor_on_device(
        input_a.shape,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- Tile count ---------------------------------------------------------
    total_tiles = prod(input_a.padded_shape) // _TILE_ELEMENTS
    assert (
        input_a.padded_shape[-1] % _TILE_W == 0 and input_a.padded_shape[-2] % _TILE_H == 0
    ), f"Padded H,W must be tile-aligned, got {input_a.padded_shape}"

    # --- Work distribution --------------------------------------------------
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    (_, core_grid, core_group_1, core_group_2, work_per_core1, work_per_core2) = ttnn.split_work_to_cores(
        all_cores, total_tiles
    )

    # --- Circular buffers (double-buffered, block_size tiles per slot) -----
    cb_total = 2 * block_size * tile_size_bytes

    def _cb_desc(cb_index: int) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=dtype,
            page_size=tile_size_bytes,
        )
        return ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )

    cb_a_desc = _cb_desc(0)  # c_0: input A
    cb_b_desc = _cb_desc(1)  # c_1: input B
    cb_out_desc = _cb_desc(2)  # c_2: output

    # --- Compile-time args --------------------------------------------------
    ta_a = ttnn.TensorAccessorArgs(input_a).get_compile_time_args()
    ta_b = ttnn.TensorAccessorArgs(input_b).get_compile_time_args()
    ta_out = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    reader_ct_args = [0, 1, block_size] + ta_a + ta_b
    writer_ct_args = [2, block_size] + ta_out
    compute_ct_args = [block_size]

    # --- Runtime args (per-core) -------------------------------------------
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    a_addr = input_a.buffer_address()
    b_addr = input_b.buffer_address()
    out_addr = output.buffer_address()

    current_tile = 0
    for core_group, work_per_core in [
        (core_group_1, work_per_core1),
        (core_group_2, work_per_core2),
    ]:
        if work_per_core == 0:
            continue
        num_blocks = (work_per_core + block_size - 1) // block_size
        for core_range in core_group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        a_addr,
                        b_addr,
                        work_per_core,  # num_tiles
                        current_tile,  # start_tile_id
                    ]
                    writer_rt_args[x][y] = [
                        out_addr,
                        work_per_core,  # num_tiles
                        current_tile,  # start_tile_id
                    ]
                    compute_rt_args[x][y] = [num_blocks]
                    current_tile += work_per_core

    assert current_tile == total_tiles, f"work distribution mismatch: {current_tile} vs {total_tiles}"

    # --- Kernel descriptors -------------------------------------------------
    _DATAFLOW = "custom_op/mul-relu-overlap-block/operation/kernels/dataflow"
    reader_path = f"{_DATAFLOW}/reader_block.cpp"
    writer_path = f"{_DATAFLOW}/writer_block.cpp"
    compute_path = "custom_op/mul-relu-overlap-block/operation/kernels/compute/compute_mul_relu_overlap_block.cpp"

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=reader_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        defines=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    reader_kernel.runtime_args = reader_rt_args

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=writer_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        defines=[],
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_kernel.runtime_args = writer_rt_args

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        defines=[],
        config=compute_cfg,
    )
    compute_kernel.runtime_args = compute_rt_args

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_a_desc, cb_b_desc, cb_out_desc],
    )

    return ttnn.generic_op([input_a, input_b, output], program)
