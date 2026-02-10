# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, NamedTuple, Tuple

import ttnn


class OpDescriptor(NamedTuple):
    """
    Simple descriptor for an op and its IO tensors

    Contains:
    - descriptor: The ProgramDescriptor for the operation
    - input_tensors: All input tensors for the op
    - output_tensors: All output tensors for the op
    - keepalive: References to keep alive (e.g. GlobalSemaphores whose L1
      addresses are baked into runtime args). Using tuple for immutability.
    """

    descriptor: "ttnn.ProgramDescriptor"
    input_tensors: List["ttnn.Tensor"]
    output_tensors: List["ttnn.Tensor"]
    keepalive: Tuple[Any, ...] = ()


__all__ = ["OpDescriptor"]
