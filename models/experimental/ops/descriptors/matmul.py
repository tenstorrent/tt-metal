# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul descriptor wrapper.

Uses the C++ matmul_descriptor() path which shares all decision logic
(transpose handling, config generation, validation, factory selection)
with ttnn.matmul().

Note: This requires that the selected matmul factory implements create_descriptor().
Not all factories have been migrated yet.
"""

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def matmul(
    input_a: "ttnn.Tensor",
    input_b: "ttnn.Tensor",
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    dtype: Optional["ttnn.DataType"] = None,
    program_config=None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    core_grid: Optional["ttnn.CoreGrid"] = None,
    output_tile=None,
) -> OpDescriptor:
    """
    Create an OpDescriptor for a matmul operation.

    Args:
        input_a: First input tensor (must be on device).
        input_b: Second input tensor (must be on device).
        transpose_a: Whether to transpose input_a.
        transpose_b: Whether to transpose input_b.
        memory_config: Output memory configuration.
        dtype: Output data type.
        program_config: Program configuration.
        compute_kernel_config: Compute kernel configuration.
        core_grid: Core grid for sharding.
        output_tile: Output tile configuration.

    Returns:
        OpDescriptor containing the program descriptor, input tensors, and output tensors.
    """
    descriptor, outputs = ttnn.matmul_descriptor(
        input_a,
        input_b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        output_tile=output_tile,
    )
    return OpDescriptor(descriptor, [input_a, input_b], outputs)


__all__ = ["matmul"]
