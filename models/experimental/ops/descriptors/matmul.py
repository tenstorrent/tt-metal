# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul operation descriptor.

Creates an OpDescriptor for matrix multiplication using the
MatmulMultiCoreReuseOptimizedProgramFactory descriptor path.
"""

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
        # core_range_set is not used by create_descriptor (core range derived from program_config),
        # but we need a valid one for the API. Use single-core to match default program config.
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    # Auto-generate a simple program config if not provided
    if program_config is None:
        program_config = _default_program_config(input_a, input_b, transpose_a, transpose_b)

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

    # Create output tensors
    output_tensors = ttnn.MatmulDeviceOperation.create_output_tensors(operation_params, tensor_args)

    # Create descriptor via the factory
    program_descriptor = ttnn.MatmulMultiCoreReuseOptimizedProgramFactory.create_descriptor(
        operation_params, tensor_args, output_tensors, core_range_set
    )

    # Build OpDescriptor
    inputs = [input_a, input_b]
    outputs = list(output_tensors)

    return OpDescriptor(program_descriptor, inputs, outputs)


def _default_program_config(
    input_a: "ttnn.Tensor",
    input_b: "ttnn.Tensor",
    transpose_a: bool,
    transpose_b: bool,
) -> "ttnn.MatmulMultiCoreReuseProgramConfig":
    """Generate a simple single-core MatmulMultiCoreReuseProgramConfig.

    Uses a single core with all tiles processed on that core.
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

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=K,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M,
        per_core_N=N,
    )
