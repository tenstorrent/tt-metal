# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, NamedTuple, Optional, Tuple

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
    - co_dispatch_group: Optional (group_id, expected_count) tuple.  When set,
      composite.launch() validates that ALL ops with this group_id are present
      in the same launch call.  Used by OpGraphBuilder to enforce that barrier-
      linked paths are co-dispatched (dispatching a subset would deadlock).
    """

    descriptor: "ttnn.ProgramDescriptor"
    input_tensors: List["ttnn.Tensor"]
    output_tensors: List["ttnn.Tensor"]
    keepalive: Tuple[Any, ...] = ()
    co_dispatch_group: Optional[Tuple[str, int]] = None


__all__ = ["OpDescriptor"]
