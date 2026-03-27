# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# minimal_binary_op: experimental binary element-wise operation via ttnn.generic_op.
# Designed as a clean testbed for exploring kernel implementation strategies.
#
# Supported scope:
#   - dtype:   bfloat16 (FPU/Matrix engine) or float32 (SFPU/Vector engine)
#   - ops:     "add" or "mul"
#   - memory:  interleaved DRAM, tile layout only
#   - no broadcasting, no sharding, no mixed dtypes

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from math import prod

import math
import ttnn

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MinimalBinaryConfig:
    """Kernel implementation strategy parameters."""

    block_size: int = 1
    """Number of tiles read/written per DRAM batch (compile-time arg).
    Reader and writer always reserve `block_size` slots; if fewer tiles remain
    in the last batch, those reads/writes are clamped but the CB push/pop still
    uses the full `block_size` (garbage tail tiles are harmless)."""

    sub_block_size: int = 1
    """Number of tiles kept in DST between tile_regs_acquire/release (CTA).
    Must divide block_size.  Limits: ≤ 8 for bfloat16 (FPU), ≤ 2 for float32
    (SFPU needs 2 DST slots per tile: one for A, one for B)."""

    use_dual_reader: bool = True
    """True  → reader handles both inputs A and B  (DO_PROCESS_INPUT1 on reader).
    False → reader handles only A; writer reads B and writes C (NoC-balanced)."""

    use_flushed_writes: bool = False
    """True → writer uses noc_async_writes_flushed() (departure-only barrier)
    instead of noc_async_write_barrier() (completion barrier)."""


# ---------------------------------------------------------------------------
# Tile size helpers
# ---------------------------------------------------------------------------

_TILE_ELEMENTS = 32 * 32
_DTYPE_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4}


def _tile_size(dtype: ttnn.DataType) -> int:
    return _DTYPE_BYTES[dtype] * _TILE_ELEMENTS


# ---------------------------------------------------------------------------
# Main operation
# ---------------------------------------------------------------------------


def minimal_binary_op(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    op_type: str,
    config: MinimalBinaryConfig = MinimalBinaryConfig(),
) -> ttnn.Tensor:
    """
    Compute element-wise `op_type` ("add" or "mul") of two tensors via
    custom kernels dispatched through `ttnn.generic_op`.

    Both tensors must be:
      - same dtype (bfloat16 or float32)
      - same shape
      - tile layout, interleaved DRAM
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    assert op_type in ("add", "mul"), f"op_type must be 'add' or 'mul', got {op_type!r}"
    assert input_a.dtype == input_b.dtype, f"Mixed dtypes not supported: {input_a.dtype} vs {input_b.dtype}"
    assert input_a.dtype in _DTYPE_BYTES, f"Only bfloat16 and float32 are supported, got {input_a.dtype}"
    assert input_a.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_b.layout == ttnn.TILE_LAYOUT, "Inputs must be in TILE_LAYOUT"
    assert input_a.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input A must be in DRAM (interleaved)"
    assert input_b.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Input B must be in DRAM (interleaved)"
    assert list(input_a.shape) == list(input_b.shape), f"Shape mismatch: {input_a.shape} vs {input_b.shape}"
    assert config.sub_block_size <= config.block_size, "sub_block_size must be <= block_size"
    assert config.block_size % config.sub_block_size == 0, "block_size must be divisible by sub_block_size"
    dtype = input_a.dtype
    is_fp32 = dtype == ttnn.float32
    if is_fp32:
        assert config.sub_block_size <= 2, "sub_block_size > 2 not supported for float32 (DST capacity)"

    device = input_a.device()

    (tile_h, tile_w) = input_a.get_tile().tile_shape
    tensor_size = math.prod(input_a.padded_shape)
    num_tiles = tensor_size / (tile_h * tile_w)
    # print(f"num tiles = {num_tiles}")

    # ------------------------------------------------------------------
    # Output tensor allocation
    # ------------------------------------------------------------------
    output = ttnn.allocate_tensor_on_device(
        input_a.shape,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # ------------------------------------------------------------------
    # Work distribution
    # ------------------------------------------------------------------
    total_tiles = prod(input_a.padded_shape) // _TILE_ELEMENTS

    # print(f"total tiles = {total_tiles}")

    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    (_, core_grid, core_group_1, core_group_2, work_per_core1, work_per_core2) = ttnn.split_work_to_cores(
        all_cores, total_tiles
    )

    num_cores = grid.x * grid.y
    # print(f"num cores = {num_cores}")

    # ------------------------------------------------------------------
    # Circular buffer setup (double-buffered)
    # ------------------------------------------------------------------
    tile_sz = _tile_size(dtype)
    cb_total = config.block_size * 2 * tile_sz  # 2× for double buffering

    def _cb_desc(cb_index: int) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=dtype,
            page_size=tile_sz,
        )
        return ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )

    cb_a_desc = _cb_desc(0)  # c_0: input A
    cb_b_desc = _cb_desc(1)  # c_1: input B
    cb_out_desc = _cb_desc(2)  # c_2: output C

    # ------------------------------------------------------------------
    # Compile-time args
    # ------------------------------------------------------------------
    ta_a = ttnn.TensorAccessorArgs(input_a).get_compile_time_args()
    ta_b = ttnn.TensorAccessorArgs(input_b).get_compile_time_args()
    ta_out = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    reader_ct_args = [config.block_size] + ta_a
    if config.use_dual_reader:
        reader_ct_args += ta_b

    writer_ct_args = [config.block_size] + ta_out
    if not config.use_dual_reader:
        writer_ct_args += ta_b

    compute_ct_args = [config.block_size, config.sub_block_size]

    # ------------------------------------------------------------------
    # Defines
    # ------------------------------------------------------------------
    reader_defines = []
    if config.use_dual_reader:
        reader_defines.append(("DO_PROCESS_INPUT1", "1"))

    writer_defines = []
    if not config.use_dual_reader:
        writer_defines.append(("DO_PROCESS_INPUT1", "1"))
    if config.use_flushed_writes:
        writer_defines.append(("USE_FLUSHED_WRITES", "1"))

    # Compute defines: IS_FP32, PROCESS_OP_INIT, PROCESS_OP
    if is_fp32:
        if op_type == "add":
            op_init = "add_binary_tile_init()"
            op_call = "add_binary_tile"
        else:
            op_init = "mul_binary_tile_init()"
            op_call = "mul_binary_tile"
    else:
        if op_type == "add":
            op_init = "binary_tiles_init<true, ELWADD>(tt::CBIndex::c_0, tt::CBIndex::c_1)"
            op_call = "add_tiles"
        else:
            op_init = "binary_tiles_init<true, ELWMUL>(tt::CBIndex::c_0, tt::CBIndex::c_1)"
            op_call = "mul_tiles"

    compute_defines = [
        ("IS_FP32", "1" if is_fp32 else "0"),
        ("PROCESS_OP_INIT", op_init),
        ("PROCESS_OP", op_call),
    ]

    # ------------------------------------------------------------------
    # Runtime args
    # ------------------------------------------------------------------
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

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
                    # print(f"core {x}, {y} - {work_per_core} tiles, start tile = {current_tile}")

                    r_args = [input_a.buffer_address(), current_tile, work_per_core]
                    if config.use_dual_reader:
                        r_args.append(input_b.buffer_address())
                    reader_rt_args[x][y] = r_args

                    w_args = [output.buffer_address(), current_tile, work_per_core]
                    if not config.use_dual_reader:
                        w_args.append(input_b.buffer_address())
                    writer_rt_args[x][y] = w_args

                    compute_rt_args[x][y] = [work_per_core]

                    current_tile += work_per_core

    assert current_tile == total_tiles
    # ------------------------------------------------------------------
    # Kernel paths (relative to repo root)
    # ------------------------------------------------------------------
    _KERNEL_BASE = "custom_op/minimal_binary/operation/kernels"
    reader_path = f"{_KERNEL_BASE}/dataflow/reader_minimal_binary.cpp"
    writer_path = f"{_KERNEL_BASE}/dataflow/writer_minimal_binary.cpp"
    compute_path = f"{_KERNEL_BASE}/compute/compute_minimal_binary.cpp"

    # ------------------------------------------------------------------
    # Kernel descriptors
    # ------------------------------------------------------------------
    # KernelDescriptor constructor does not accept RuntimeArgs wrapper directly;
    # runtime_args must be set as a property after construction.
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=reader_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        defines=reader_defines,
        config=ttnn.ReaderConfigDescriptor(),
    )
    reader_kernel.runtime_args = reader_rt_args

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=writer_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        defines=writer_defines,
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_kernel.runtime_args = writer_rt_args

    import ttnn._ttnn.program_descriptor as _pd

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=is_fp32,
    )
    if is_fp32:
        # Unpack both input CBs to full float32 precision in DST
        # (same as binary_ng SFPU path: unpack_to_dest_mode = UnpackToDestFp32 for c_0 and c_1)
        _NUM_CBS = 64
        modes = _pd.VectorUnpackToDestMode()
        for _ in range(_NUM_CBS):
            modes.append(ttnn.UnpackToDestMode.Default)
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32  # c_0: input A
        modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32  # c_1: input B
        compute_cfg.unpack_to_dest_mode = modes

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        defines=compute_defines,
        config=compute_cfg,
    )
    compute_kernel.runtime_args = compute_rt_args

    # ------------------------------------------------------------------
    # Program descriptor and dispatch
    # ------------------------------------------------------------------
    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_a_desc, cb_b_desc, cb_out_desc],
    )

    return ttnn.generic_op([input_a, input_b, output], program)
