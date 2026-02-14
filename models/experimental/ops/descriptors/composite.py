# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def _validate_co_dispatch_groups(op_descriptors: List[OpDescriptor]) -> None:
    """Validate that all members of each co-dispatch group are present.

    OpGraph paths share barrier semaphores and MUST be dispatched together.
    Dispatching a subset would deadlock because missing paths' cores never
    arrive at the barrier.

    Raises:
        ValueError: If a co-dispatch group is incomplete.
    """
    groups: Dict[str, Tuple[int, int]] = {}  # group_id -> (expected, actual_count)
    for op in op_descriptors:
        if op.co_dispatch_group is not None:
            gid, expected = op.co_dispatch_group
            if gid not in groups:
                groups[gid] = (expected, 0)
            groups[gid] = (groups[gid][0], groups[gid][1] + 1)

    for gid, (expected, actual) in groups.items():
        if actual != expected:
            raise ValueError(
                f"Co-dispatch group '{gid}': expected {expected} ops but got {actual}. "
                f"All OpGraph paths must be dispatched together via composite.launch()."
            )


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
        >>> left_out, right_out = launch([left, right])
    """
    if not op_descriptors:
        raise ValueError("op_descriptors cannot be empty")

    # Validate co-dispatch groups: all members of a group must be present
    _validate_co_dispatch_groups(op_descriptors)

    # Extract program descriptors
    descriptors = [op_descriptor.descriptor for op_descriptor in op_descriptors]

    # Merge descriptors (or use single descriptor directly)
    if len(descriptors) == 1:
        merged = descriptors[0]
    else:
        # Merge all descriptors - validates no overlapping core ranges
        merged = ttnn.merge_program_descriptors(descriptors)

    # Build io_tensors list
    io_tensors = [t for op_descriptor in op_descriptors for t in op_descriptor.input_tensors] + [
        t for op_descriptor in op_descriptors for t in op_descriptor.output_tensors
    ]

    # Execute the merged program
    ttnn.generic_op(io_tensors, merged)

    # Return output tensors from each op
    return [op_descriptor.output_tensors for op_descriptor in op_descriptors]


__all__ = ["launch"]
