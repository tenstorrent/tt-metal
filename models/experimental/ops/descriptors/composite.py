# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def launch(op_descriptors: List[OpDescriptor]) -> List[List["ttnn.Tensor"]]:
    """
    Launch a composite operation from multiple op descriptors.

    Merges the program descriptors from each op descriptor and executes them as a single
    program. Each op descriptor operates on non-overlapping cores.

    The device program cache (enabled via device.enable_program_cache()) uses
    GenericOpDeviceOperation::compute_program_hash to cache the compiled program,
    so subsequent calls with structurally identical descriptors reuse the compiled
    program automatically.

    Args:
        op_descriptors: A list of OpDescriptor objects

    Returns:
        A list of output tensors, one per op descriptor, in the same order as the input op descriptors.

    Raises:
        ValueError: If op_descriptors is empty.
        RuntimeError: If any op descriptors have overlapping core ranges.

    Example:
        >>> left = models.experimental.ops.descriptors.normalization.rms_norm(input1, weight=w1, cores=cores1)
        >>> right = models.experimental.ops.descriptors.normalization.rms_norm(input2, weight=w2, cores=cores2)
        >>> for _ in range(100):
        ...     left_out, right_out = launch([left, right])
        >>> assert left_out is left.output_tensors[0]
    """
    if not op_descriptors:
        raise ValueError("op_descriptors cannot be empty")

    # Extract program descriptors
    descriptors = [op_descriptor.descriptor for op_descriptor in op_descriptors]

    # Merge descriptors (or use single descriptor directly)
    if len(descriptors) == 1:
        merged = descriptors[0]
    else:
        # Merge all descriptors - validates no overlapping core ranges
        merged = ttnn.ProgramDescriptor.merge_descriptors(descriptors)

    # Build io_tensors list
    io_tensors = [t for op_descriptor in op_descriptors for t in op_descriptor.input_tensors] + [
        t for op_descriptor in op_descriptors for t in op_descriptor.output_tensors
    ]

    # Execute the merged program
    # Device program cache (via compute_program_hash) handles compiled program caching
    ttnn.generic_op(io_tensors, merged)

    # Return output tensors from each op descriptor (first output for each, matching ProgramBranch.output)
    return [op_descriptor.output_tensors for op_descriptor in op_descriptors]


__all__ = ["launch"]
