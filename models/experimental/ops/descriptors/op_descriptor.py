# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, NamedTuple, Tuple

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


class FusedOp:
    """Result of fusing ops via Sequential/Parallel.

    Wraps an ``OpDescriptor`` and adds ``semaphores`` refs that prevent
    GC of GlobalSemaphores whose L1 addresses are baked into runtime args.

    Properties ``descriptor``, ``input_tensors``, and ``output_tensors``
    forward to the underlying ``OpDescriptor``, so ``FusedOp`` is
    duck-type compatible with ``OpDescriptor`` (e.g. for
    ``composite.launch()``).

    Cannot be nested in Sequential/Parallel -- ``_resolve()`` rejects it
    with a TypeError.
    """

    __slots__ = ("op", "semaphores")

    def __init__(
        self,
        op: OpDescriptor,
        semaphores: Tuple[Any, ...] = (),
    ):
        self.op = op
        self.semaphores = semaphores

    @property
    def descriptor(self):
        return self.op.descriptor

    @property
    def input_tensors(self):
        return self.op.input_tensors

    @property
    def output_tensors(self):
        return self.op.output_tensors

    def __repr__(self):
        n_kernels = len(self.op.descriptor.kernels) if hasattr(self.op.descriptor, "kernels") else "?"
        return (
            f"FusedOp(kernels={n_kernels}, "
            f"inputs={len(self.op.input_tensors)}, "
            f"outputs={len(self.op.output_tensors)})"
        )


__all__ = ["OpDescriptor", "FusedOp"]
