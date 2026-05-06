# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# mul_relu_add_overlap_op: experimental fused y = relu(a * b) + c via ttnn.generic_op.
#
# Naive sequential baseline (no FPU/SFPU overlap): mul_tiles (FPU/MATH) -> relu_tile
# (SFPU/MATH) -> add via DEST->SrcA reuse (FPU/MATH), all on TRISC1. A future
# iteration will move relu to PACK (TRISC2) to overlap with the surrounding
# FPU ops.
#
# Supported scope:
#   - dtype:   bfloat16 only
#   - memory:  interleaved DRAM, tile layout
#   - shapes:  same shape for a, b, c, output (no broadcast, no sharding)
#
# Dataflow kernels are reused (path reference) from
# ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/.

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import ttnn


_TILE_H = 32
_TILE_W = 32
_TILE_ELEMENTS = _TILE_H * _TILE_W
_BF16_BYTES = 2
_TILE_SIZE_BYTES = _TILE_ELEMENTS * _BF16_BYTES  # 2048


@dataclass
class MulReluAddOverlapConfig:
    """Placeholder for future overlap toggles. Empty for now."""

    pass


def _decompose_shape(padded_shape):
    """
    Map an arbitrary-rank padded shape to the (D, N, C, Ht, Wt, cND) layout
    expected by the dataflow kernels. Dims beyond rank 5 are collapsed into
    cND. H and W are converted to tile counts.
    """
    s = list(padded_shape)
    while len(s) < 5:
        s = [1] + s
    *nd_dims, D, N, C, H, W = s
    assert H % _TILE_H == 0 and W % _TILE_W == 0, f"Padded H,W must be tile-aligned, got {H}x{W}"
    Ht, Wt = H // _TILE_H, W // _TILE_W
    cND = 1
    for d in nd_dims:
        cND *= d
    return D, N, C, Ht, Wt, cND


def mul_relu_add_overlap_op(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    input_c: ttnn.Tensor,
    config: MulReluAddOverlapConfig = MulReluAddOverlapConfig(),
) -> ttnn.Tensor:
    """y = relu(a * b) + c, elementwise, bfloat16, DRAM-interleaved tile layout."""
    # --- Validation ---------------------------------------------------------
    assert input_a.dtype == ttnn.bfloat16, f"Only bfloat16 supported, got {input_a.dtype}"
    assert input_b.dtype == ttnn.bfloat16, f"Only bfloat16 supported, got {input_b.dtype}"
    assert input_c.dtype == ttnn.bfloat16, f"Only bfloat16 supported, got {input_c.dtype}"
    assert input_a.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_b.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_c.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_a.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input A must be in DRAM (interleaved)"
    assert input_b.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input B must be in DRAM (interleaved)"
    assert input_c.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input C must be in DRAM (interleaved)"
    assert list(input_a.shape) == list(input_b.shape), f"Shape mismatch a vs b: {input_a.shape} vs {input_b.shape}"
    assert list(input_a.shape) == list(input_c.shape), f"Shape mismatch a vs c: {input_a.shape} vs {input_c.shape}"

    device = input_a.device()

    # --- Output tensor ------------------------------------------------------
    output = ttnn.allocate_tensor_on_device(
        input_a.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- Shape decomposition for dataflow kernels --------------------------
    D, N, C, Ht, Wt, cND = _decompose_shape(input_a.padded_shape)
    total_tiles = prod(input_a.padded_shape) // _TILE_ELEMENTS
    assert total_tiles == D * N * C * Ht * Wt * cND, f"tile count mismatch: {total_tiles} vs {D*N*C*Ht*Wt*cND}"

    # --- Work distribution --------------------------------------------------
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    (_, core_grid, core_group_1, core_group_2, work_per_core1, work_per_core2) = ttnn.split_work_to_cores(
        all_cores, total_tiles
    )

    # --- Circular buffers (double-buffered, single tile each) --------------
    cb_total = 2 * _TILE_SIZE_BYTES

    def _cb_desc(cb_index: int) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=ttnn.bfloat16,
            page_size=_TILE_SIZE_BYTES,
        )
        return ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )

    cb_a_desc = _cb_desc(0)  # c_0: input A
    cb_b_desc = _cb_desc(1)  # c_1: input B
    cb_c_desc = _cb_desc(2)  # c_2: input C
    cb_out_desc = _cb_desc(3)  # c_3: output

    # --- Compile-time args (match ternary kernel layout) -------------------
    ta_a = ttnn.TensorAccessorArgs(input_a).get_compile_time_args()
    ta_b = ttnn.TensorAccessorArgs(input_b).get_compile_time_args()
    ta_c = ttnn.TensorAccessorArgs(input_c).get_compile_time_args()
    ta_out = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    # ternary_reader_nobcast_ttt.cpp expects: [cb_a_idx, cb_b_idx, cb_c_idx]
    # followed by chained TensorAccessorArgs for a, b, c.
    reader_ct_args = [0, 1, 2] + ta_a + ta_b + ta_c
    # ternary_writer_nobcast.cpp expects: [cb_out_idx] + TensorAccessorArgs(out)
    # followed by [has_sharding] flag.
    has_sharding = 0
    writer_ct_args = [3] + ta_out + [has_sharding]
    compute_ct_args = []  # nothing — tile count goes in runtime args

    # --- Runtime args (per-core) -------------------------------------------
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    a_addr = input_a.buffer_address()
    b_addr = input_b.buffer_address()
    c_addr = input_c.buffer_address()
    out_addr = output.buffer_address()

    current_tile = 0
    for core_group, work_per_core in [
        (core_group_1, work_per_core1),
        (core_group_2, work_per_core2),
    ]:
        if work_per_core == 0:
            continue
        for core_range in core_group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    # Reader (5 args, matches ternary_reader_nobcast_ttt.cpp:12-16).
                    reader_rt_args[x][y] = [
                        a_addr,
                        b_addr,
                        c_addr,
                        work_per_core,  # num_tiles
                        current_tile,  # start_id
                    ]
                    # Writer (10 args, matches ternary_writer_nobcast.cpp:13-23).
                    # Note: arg order differs from binary_ng — num_tiles before start_tile_id.
                    writer_rt_args[x][y] = [
                        out_addr,
                        work_per_core,  # dst_num_tiles
                        current_tile,  # start_tile_id
                        0,  # dst_shard_width
                        D,
                        N,
                        C,
                        Ht,
                        Wt,
                        cND,
                    ]
                    compute_rt_args[x][y] = [work_per_core]
                    current_tile += work_per_core

    assert current_tile == total_tiles, f"work distribution mismatch: {current_tile} vs {total_tiles}"

    # --- Kernel descriptors -------------------------------------------------
    # Dataflow kernels referenced directly from ttnn ternary ops (no copy).
    _TERNARY_DATAFLOW = "ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow"
    reader_path = f"{_TERNARY_DATAFLOW}/ternary_reader_nobcast_ttt.cpp"
    writer_path = f"{_TERNARY_DATAFLOW}/ternary_writer_nobcast.cpp"
    compute_path = "custom_op/mul-relu-add-overlap/operation/kernels/compute/compute_mul_relu_add_overlap.cpp"

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
        fp32_dest_acc_en=True,
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
        cbs=[cb_a_desc, cb_b_desc, cb_c_desc, cb_out_desc],
    )

    return ttnn.generic_op([input_a, input_b, input_c, output], program)
