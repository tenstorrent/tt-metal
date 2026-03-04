# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def slice(
    input_tensor: "ttnn.Tensor",
    begins: List[int],
    ends: List[int],
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> OpDescriptor:
    """Create a slice OpDescriptor for tiled tensors.

    Args:
        input_tensor: Input tensor (must be on device, tiled layout).
        begins: Start indices for slicing (e.g. [0, 0, 0, 0]).
        ends: End indices for slicing (e.g. [1, 1, 256, 1024]).
        core_range_set: Optional core range set to restrict execution.
        memory_config: Optional output memory configuration.
            Defaults to input's memory config.

    Returns:
        OpDescriptor containing the program descriptor, input tensors,
        and output tensors.
    """
    device = input_tensor.device()

    if memory_config is None:
        memory_config = input_tensor.memory_config()

    rank = len(input_tensor.shape)
    step = [1] * rank

    params = ttnn.SliceParams()
    params.slice_start = ttnn.Shape(begins)
    params.slice_end = ttnn.Shape(ends)
    params.step = ttnn.Shape(step)
    params.output_mem_config = memory_config
    if core_range_set is not None:
        params.sub_core_grids = core_range_set

    tensor_args = ttnn.SliceInputs()
    tensor_args.input = input_tensor

    output_tensor = ttnn.SliceDeviceOperation.create_output_tensors(params, tensor_args)

    descriptor = ttnn.SliceTileProgramFactory.create_descriptor(params, tensor_args, output_tensor)

    return OpDescriptor(
        descriptor=descriptor,
        input_tensors=[input_tensor],
        output_tensors=[output_tensor],
        name="slice",
    )


__all__ = ["slice"]
