# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, NamedTuple

import ttnn


class OpDescriptor(NamedTuple):
    """
    Simple descriptor for an op and its IO tensors.

    Contains:
    - descriptor: The ProgramDescriptor for the operation
    - input_tensors: All input tensors for the op
    - output_tensors: All output tensors for the op
    """

    descriptor: "ttnn.ProgramDescriptor"
    input_tensors: List["ttnn.Tensor"]
    output_tensors: List["ttnn.Tensor"]


# FusedOp moved to models.experimental.ops.descriptors.fusion.fusion


__all__ = ["OpDescriptor"]
