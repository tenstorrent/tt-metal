# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-core Tilize 16x32 operation.

This implements tilize operation on a single core where:
- Input: [16, N] in row-major format (HEIGHT_SHARDED)
- Output: [16, N] in tiled format (HEIGHT_SHARDED), tiled into 16x32 blocks
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def golden(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Golden reference implementation of tilize 16x32 operation.

    Since ttnn.to_torch() automatically untilizes tiled tensors back to row-major,
    the output after tilize + untilize should be identical to the input.

    Args:
        input_tensor: Input tensor [16, N] in row-major format (N must be divisible by 64)
        (odd tile dimensions are not supported yet for fast tilize 16xN)

    Returns:
        Reference output tensor [16, N] in row-major format (same as input)
    """
    H, W = input_tensor.shape
    assert H == 16, f"Expected height=16, got {H}"
    assert W % 64 == 0, f"Width must be divisible by 64, got {W}"

    # Tilize + untilize is a reversible operation, so output equals input
    return input_tensor.clone()


def tilize_16x32_kernel(input_tensor, output_tensor):
    """
    Execute tilize 16x32 operation using generic_op.

    Args:
        input_tensor: Input tensor [16, N] in row-major format (must be HEIGHT_SHARDED)
        output_tensor: Pre-allocated output tensor [16, N] (must be HEIGHT_SHARDED)

    Returns:
        Output tensor with tilized data
    """
    # Get shard spec from input tensor
    shard_spec = input_tensor.memory_config().shard_spec
    shard_shape = shard_spec.shape

    # Get core grid from shard spec
    core_grid = shard_spec.grid

    # Calculate dimensions
    # Input is [16, N] where N should be divisible by 32
    H = shard_shape[0]  # Should be 16
    W = shard_shape[1]  # N (256 or 64)

    # Use explicit ValueError instead of assert (which can be stripped with -O optimization)
    if H != 16:
        raise ValueError(f"Expected height=16, got {H}")
    if W % 32 != 0:
        raise ValueError(f"Width must be divisible by 32, got {W}")

    # Calculate number of 16x32 tiles along width (BLOCK_CT_DIM in fast_tilize_test.cpp)
    block_ct_dim = W // 32

    # CB indices
    in_cb = 0
    out_cb = 16  # Output operands start at 16

    # ========================================================================
    # Circular Buffer Descriptors
    # ========================================================================

    # CB 0: Input (sharded tensor, row-major format)
    input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in_cb, input_tensor)

    # CB 16: Output (sharded tensor, tiled format)
    output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

    # ========================================================================
    # Unified Kernel Descriptor (handles NCRISC, BRISC, TRISC)
    # ========================================================================

    # Named compile-time args for NCRISC (reader - no-op for tilize)
    ncrisc_named_compile_time_args = []

    # Named compile-time args for BRISC (writer - no-op for tilize)
    brisc_named_compile_time_args = []

    # Named compile-time args for TRISC (compute) - match fast_tilize_test.cpp BLOCK_CT_DIM
    trisc_named_compile_time_args = [
        ("in_cb", in_cb),
        ("out_cb", out_cb),
        ("block_ct_dim", block_ct_dim),
    ]

    # Unified kernel descriptor
    unified_kernel = UnifiedKernelDescriptor(
        kernel_source="models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/tilize_16x32/kernels/tilize_16x32_kernel.cpp",
        core_ranges=core_grid,
        ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
        brisc_named_compile_time_args=brisc_named_compile_time_args,
        trisc_named_compile_time_args=trisc_named_compile_time_args,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_active_core",
                core_range=core_grid,
                value=1,
                other_value=0,
            ),
        ],
    )

    # ========================================================================
    # Program Descriptor
    # ========================================================================
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[
            input_cb_descriptor,
            output_cb_descriptor,
        ],
    )

    # Execute generic op
    io_tensors = [input_tensor, output_tensor]
    output = ttnn.generic_op(io_tensors, program_descriptor)

    return output
