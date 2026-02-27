# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, NamedTuple, Optional

import ttnn


class OpDescriptor(NamedTuple):
    """
    Simple descriptor for an op and its IO tensors.

    Contains:
    - descriptor: The ProgramDescriptor for the operation
    - input_tensors: All input tensors for the op
    - output_tensors: All output tensors for the op
    - allowed_core_range: Optional CoreRangeSet for topology validation.
        When set, topology validation uses this range instead of the
        actual kernel core ranges. This allows a parent op (e.g. sharded
        LN on 4 cores) to declare a wider region (e.g. 32 cores) that
        its subtree is allowed to use.
    """

    descriptor: "ttnn.ProgramDescriptor"
    input_tensors: List["ttnn.Tensor"]
    output_tensors: List["ttnn.Tensor"]
    name: str = ""
    allowed_core_range: Optional[Any] = None


# FusedOp moved to models.experimental.ops.descriptors.fusion.fusion


__all__ = ["OpDescriptor"]
