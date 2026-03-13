# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul operation descriptor.

Creates an OpDescriptor for matrix multiplication using the
MatmulMultiCoreReuseOptimizedProgramFactory descriptor path.

WARNING: This descriptor always uses MatmulMultiCoreReuseOptimizedProgramFactory.
The ttnn.matmul() path may select a different factory (e.g. multicast) for the
same shapes.
Further, it computes a simplified program config that does not guarantee parity
with the ttnn.matmul() path. This descriptor is currently for demonstration purposes only.
When all variants of matmul have a create_descriptor(), and have the ability
to construct a program config from a CoreRangeSet so it can be offset from core 0,0,
then this descriptor interface can be made general.
"""

import math
from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def matmul(
    input_a: "ttnn.Tensor",
    input_b: "ttnn.Tensor",
    *,
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    program_config: Optional["ttnn.MatmulMultiCoreReuseProgramConfig"] = None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    output_mem_config: Optional["ttnn.MemoryConfig"] = None,
    output_dtype: Optional["ttnn.DataType"] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> "OpDescriptor":
    """Create a matmul op descriptor.

    Args:
        input_a: First input tensor (on device, tiled).
        input_b: Second input tensor (on device, tiled).
        core_range_set: Optional core range set override.
        program_config: MatmulMultiCoreReuseProgramConfig. If None, a simple
            single-core config is generated from tensor shapes.
        compute_kernel_config: Compute kernel configuration.
        output_mem_config: Output memory configuration.
        output_dtype: Output data type. Defaults to input_a dtype.
        transpose_a: Transpose first input.
        transpose_b: Transpose second input.

    Returns:
        OpDescriptor with matmul program descriptor and IO tensors.
    """
    device = input_a.device()

    if core_range_set is None:
        grid_size = device.compute_with_storage_grid_size()
        core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
        )

    # Auto-generate a simple program config if not provided
    if program_config is None:
        fp32 = getattr(compute_kernel_config, "fp32_dest_acc_en", False) if compute_kernel_config else False
        program_config = _default_program_config(input_a, input_b, transpose_a, transpose_b, core_range_set, fp32)

    # Use create_matmul_attributes to finalize params (computes bcast_batch, output_dtype, etc.)
    base_params = ttnn.MatmulParams()
    base_params.program_config = program_config
    base_params.transpose_a = transpose_a
    base_params.transpose_b = transpose_b
    if output_dtype is not None:
        base_params.output_dtype = output_dtype
    if output_mem_config is not None:
        base_params.output_mem_config = output_mem_config
    if compute_kernel_config is not None:
        base_params.compute_kernel_config = compute_kernel_config

    operation_params = ttnn.create_matmul_attributes(input_a, input_b, base_params, [])

    # Build MatmulInputs
    tensor_args = ttnn.MatmulInputs()
    tensor_args.input_tensors = [input_a, input_b]

    # Get the output spec from the factory (shape, dtype, tile, shard shape, etc.).
    # When the output is sharded, compute_output_specs() computes the correct shard
    # shape/orientation but places it on a (0,0)-based grid. Replace the grid with
    # core_range_set, which is the single source of truth for where this op lives.
    output_spec = ttnn.MatmulDeviceOperation.compute_output_specs(operation_params, tensor_args)[0]

    if output_mem_config is not None and output_mem_config.is_sharded():
        factory_shard = output_spec.memory_config.shard_spec
        corrected_shard = ttnn.ShardSpec(core_range_set, factory_shard.shape, factory_shard.orientation)
        output_spec = ttnn.TensorSpec(
            output_spec.shape,
            output_spec.dtype,
            output_spec.layout,
            output_mem_config.memory_layout,
            corrected_shard,
            output_mem_config.buffer_type,
            output_spec.tile,
        )

    output_tensors = [ttnn.allocate_tensor_on_device(output_spec, device)]

    # Create descriptor via the factory.
    # Only MatmulMultiCoreReuseOptimizedProgramFactory is supported for now.
    program_descriptor = ttnn.MatmulMultiCoreReuseOptimizedProgramFactory.create_descriptor(
        operation_params, tensor_args, output_tensors, core_range_set
    )

    # Build OpDescriptor
    inputs = [input_a, input_b]
    outputs = list(output_tensors)

    return OpDescriptor(program_descriptor, inputs, outputs, "matmul")


def _default_program_config(
    input_a: "ttnn.Tensor",
    input_b: "ttnn.Tensor",
    transpose_a: bool,
    transpose_b: bool,
    core_range_set: "ttnn.CoreRangeSet",
    fp32_dest_acc_en: bool,
) -> "ttnn.MatmulMultiCoreReuseProgramConfig":
    """Generate a MatmulMultiCoreReuseProgramConfig from tensor shapes and core grid.

    Distributes M rows across cores. Each core handles all N columns.
    """
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    a_shape = input_a.padded_shape
    b_shape = input_b.padded_shape

    if transpose_a:
        M = a_shape[-1] // TILE_WIDTH
        K = a_shape[-2] // TILE_HEIGHT
    else:
        M = a_shape[-2] // TILE_HEIGHT
        K = a_shape[-1] // TILE_WIDTH

    if transpose_b:
        N = b_shape[-2] // TILE_HEIGHT
    else:
        N = b_shape[-1] // TILE_WIDTH

    # Count cores and compute bounding box from the core range set
    num_cores = 0
    max_x = 0
    max_y = 0
    for cr in core_range_set.ranges():
        start = cr.start
        end = cr.end
        num_cores += (end.x - start.x + 1) * (end.y - start.y + 1)
        max_x = max(max_x, end.x)
        max_y = max(max_y, end.y)

    grid_size = ttnn.CoreCoord(max_x + 1, max_y + 1)

    per_core_M = math.ceil(M / num_cores)
    per_core_N = N
    in0_block_w = K

    # Subblock tiling: maximize out_subblock_w under fp32/fp16 limit
    limit = 4 if fp32_dest_acc_en else 8
    out_subblock_w = min(per_core_N, limit)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h = 1

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )
