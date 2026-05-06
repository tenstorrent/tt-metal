# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# mul_relu_overlap_op: experimental fused y = relu(a * b) op via ttnn.generic_op.
#
# Baseline (no FPU/SFPU overlap yet): both mul_tiles (FPU) and relu_tile (SFPU)
# run sequentially on MATH (TRISC1). A future iteration will move relu to PACK
# (TRISC2) to overlap with the next tile's FPU mul.
#
# Supported scope:
#   - dtype:   bfloat16 only
#   - memory:  interleaved DRAM, tile layout
#   - shapes:  same shape for a, b, output (no broadcast, no sharding)
#
# Dataflow kernels are reused (verbatim copies) from
# ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/.
# Runtime args follow the no-broadcast recipe in binary_ng_program_factory.cpp.

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
class MulReluOverlapConfig:
    """Placeholder for future overlap toggles. Empty for now."""

    pass


def _decompose_shape(padded_shape):
    """
    Map an arbitrary-rank padded shape to the (D, N, C, Ht, Wt, cND) layout
    expected by the binary_ng dataflow kernels. Dims beyond rank 5 are
    collapsed into cND. H and W are converted to tile counts.
    """
    s = list(padded_shape)
    # Pad with leading 1s up to at least rank 5.
    while len(s) < 5:
        s = [1] + s
    *nd_dims, D, N, C, H, W = s
    assert H % _TILE_H == 0 and W % _TILE_W == 0, f"Padded H,W must be tile-aligned, got {H}x{W}"
    Ht, Wt = H // _TILE_H, W // _TILE_W
    cND = 1
    for d in nd_dims:
        cND *= d
    return D, N, C, Ht, Wt, cND


def mul_relu_overlap_op(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    config: MulReluOverlapConfig = MulReluOverlapConfig(),
) -> ttnn.Tensor:
    """y = relu(a * b), elementwise, bfloat16, DRAM-interleaved tile layout."""
    # --- Validation ---------------------------------------------------------
    assert input_a.dtype == ttnn.bfloat16, f"Only bfloat16 supported, got {input_a.dtype}"
    assert input_b.dtype == ttnn.bfloat16, f"Only bfloat16 supported, got {input_b.dtype}"
    assert input_a.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_b.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_a.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input A must be in DRAM (interleaved)"
    assert input_b.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input B must be in DRAM (interleaved)"
    assert list(input_a.shape) == list(input_b.shape), f"Shape mismatch: {input_a.shape} vs {input_b.shape}"

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

    # Strides (same for a, b, c since shapes are identical, no broadcast).
    nD_stride = Ht * Wt * C * N * D * (cND > 1)
    d_stride = Ht * Wt * C * N * (D > 1)
    n_stride = Ht * Wt * C * (N > 1)
    c_stride = Ht * Wt * (C > 1)

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
    cb_out_desc = _cb_desc(2)  # c_2: output

    # --- Compile-time args (match binary_ng kernel layout) ------------------
    ta_a = ttnn.TensorAccessorArgs(input_a).get_compile_time_args()
    ta_b = ttnn.TensorAccessorArgs(input_b).get_compile_time_args()
    ta_out = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    has_sharding = 0
    reader_ct_args = ta_a + ta_b + [has_sharding]
    writer_ct_args = ta_out + [has_sharding]
    compute_ct_args = []  # nothing — tile count goes in runtime args

    # --- Runtime args (per-core) -------------------------------------------
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    a_addr = input_a.buffer_address()
    b_addr = input_b.buffer_address()
    c_addr = output.buffer_address()

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
                    # Reader (21 args, matches binary_ng_program_factory.cpp:1179-1201).
                    reader_rt_args[x][y] = [
                        a_addr,
                        current_tile,  # start_tile_id
                        work_per_core,  # src_num_tiles (a)
                        work_per_core,  # dst_num_tiles
                        0,  # dst_shard_width (not sharded)
                        nD_stride,
                        d_stride,
                        n_stride,
                        c_stride,  # a strides
                        D,
                        N,
                        C,
                        Ht,
                        Wt,
                        cND,
                        b_addr,
                        nD_stride,
                        d_stride,
                        n_stride,
                        c_stride,  # b strides (= a's)
                        work_per_core,  # src_num_tiles_b
                    ]
                    # Writer (10 args + trailing pad, matches lines 1072-1083).
                    writer_rt_args[x][y] = [
                        c_addr,
                        current_tile,  # start_tile_id
                        work_per_core,  # dst_num_tiles
                        0,  # dst_shard_width
                        D,
                        N,
                        C,
                        Ht,
                        Wt,
                        cND,
                        0,  # trailing pad
                    ]
                    compute_rt_args[x][y] = [work_per_core]
                    current_tile += work_per_core

    assert current_tile == total_tiles, f"work distribution mismatch: {current_tile} vs {total_tiles}"

    # --- Kernel descriptors -------------------------------------------------
    # Dataflow kernels are reused directly from binary_ng (kernels_ng version):
    # they already accept two interleaved-DRAM tile-layout inputs and one output,
    # one tile at a time. Compute kernel is custom (mul + relu fused).
    _BINARY_NG_DATAFLOW = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow"
    reader_path = f"{_BINARY_NG_DATAFLOW}/reader_interleaved_no_bcast.cpp"
    writer_path = f"{_BINARY_NG_DATAFLOW}/writer_interleaved_no_bcast.cpp"
    compute_path = "custom_op/mul-relu-overlap/operation/kernels/compute/compute_mul_relu_overlap.cpp"

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
        fp32_dest_acc_en=False,
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
