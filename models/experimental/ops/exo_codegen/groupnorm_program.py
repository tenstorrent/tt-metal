# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Program builder for Exo-generated GroupNorm.

Assembles a ProgramDescriptor from the Exo-generated compute kernel and
custom reader/writer kernels. Handles CB setup, compile-time args, and
runtime args for single-core GroupNorm execution.

Simplified config for PoC:
    - Single core (1x1 grid)
    - No gamma/beta (identity affine transform)
    - Input mask generated on-chip (all 1.0, aligned groups)
    - No TILIZE_IN / UNTILIZE_OUT
    - block_w = 1 (single tile column)
"""

from __future__ import annotations

import struct

import ttnn

from models.experimental.ops.exo_codegen.groupnorm_codegen import (
    generate_groupnorm_kernels,
)


def _pack_bfloat16(value: float) -> int:
    """Pack a float into a uint32 with bfloat16 value doubled.

    The generate_reduce_scaler() and generate_bcast_col_scalar() functions
    expect a uint32_t where the bfloat16 value is packed in both halves
    (high 16 bits and low 16 bits).
    """
    # Convert float to IEEE 754 binary32
    float_bytes = struct.pack(">f", value)
    # BFloat16 is the upper 16 bits of float32
    bf16 = int.from_bytes(float_bytes[:2], "big")
    # Double-pack: same value in high and low 16 bits
    return (bf16 << 16) | bf16


def build_groupnorm_program(
    device,
    input_tensor,
    output_tensor,
    num_groups: int = 1,
    eps: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for GroupNorm using Exo-generated kernels.

    Args:
        device: TT device.
        input_tensor: Input tensor (on device, TILE_LAYOUT, BFloat16).
        output_tensor: Output tensor (on device, TILE_LAYOUT, BFloat16).
        num_groups: Number of groups (must be 1 for this PoC).
        eps: Epsilon for numerical stability.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op().
    """
    if num_groups != 1:
        raise ValueError("Exo GroupNorm PoC only supports num_groups=1")

    # Calculate dimensions
    shape = input_tensor.shape
    # GroupNorm treats input as (N, 1, C*H, W)
    num_tiles = input_tensor.volume() // (32 * 32)
    block_hw = num_tiles  # All tiles in one block
    total_elements = input_tensor.volume()

    # Single core (1x1 grid)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Generate kernel sources
    reader_src, compute_src, writer_src = generate_groupnorm_kernels()

    # --- CB setup ---
    cb_data_format = ttnn.bfloat16
    tile_size = 2 * 1024  # BFloat16 tile = 2048 bytes
    data_tile_count = block_hw

    # Map of (cb_index, num_tiles) for each circular buffer
    cb_configs = {
        0: data_tile_count,  # cb_in0: input (persistent, read 3x)
        2: 1,  # cb_scaler: reduce scaler (1.0)
        3: 1,  # cb_eps: epsilon
        4: 1,  # cb_scaler_global: 1/N
        8: 1,  # cb_ex_partial: local mean partial
        9: 1,  # cb_ex (unused but reserve for safety)
        14: 1,  # cb_ex2_global: variance
        15: 1,  # cb_ex_global: mean
        16: data_tile_count,  # cb_out0: output
        21: 1,  # cb_ex2_partial: local variance partial
        24: data_tile_count,  # cb_x: intermediate
        25: data_tile_count,  # cb_xmm: intermediate
        27: 1,  # cb_ex2pe: inv_std
        28: 1,  # cb_input_mask: mask (all 1.0)
    }

    cb_descriptors = []
    for cb_idx, num_tiles in cb_configs.items():
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_idx,
            data_format=cb_data_format,
            page_size=tile_size,
        )
        desc = ttnn.CBDescriptor(
            total_size=num_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )
        cb_descriptors.append(desc)

    # --- Compile-time args ---
    reader_ct_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    compute_ct_args = [block_hw]

    # Writer uses TensorAccessorArgs for output (at index 1 in io_tensors)
    writer_ct_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # --- Runtime args ---
    # Pack scalar values as bfloat16 pairs
    packed_scaler = _pack_bfloat16(1.0)
    packed_scaler_global = _pack_bfloat16(1.0 / total_elements)
    packed_eps = _pack_bfloat16(eps)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [
        input_tensor.buffer_address(),
        block_hw,
        0,  # start_id
    ]

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[0][0] = [
        packed_scaler,
        packed_scaler_global,
        packed_eps,
        output_tensor.buffer_address(),
        block_hw,
        0,  # start_id
    ]

    # --- Kernel descriptors ---
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
        cbs=cb_descriptors,
    )
