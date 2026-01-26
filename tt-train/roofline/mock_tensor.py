# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MockTensor with autograd-like gradient tracking for roofline modeling.

This module provides a MockTensor class that stores metadata (shape, dtype)
without actual data, while supporting autograd-like backward graph construction
for performance estimation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING
import math

from .hardware import DataType

if TYPE_CHECKING:
    from .estimate import RooflineContext


@dataclass
class BackwardNode:
    """Node in the backward computation graph.

    Each node stores a backward function and references to input tensors,
    enabling traversal of the computation graph during backward pass estimation.
    """

    backward_fn: Callable[["RooflineContext"], None]
    inputs: List["MockTensor"] = field(default_factory=list)


class MockTensor:
    """Mock tensor with autograd-like gradient tracking.

    This class mimics ttml.autograd.Tensor interface for roofline estimation,
    storing only shape/dtype metadata without allocating actual data.

    Example:
        >>> x = MockTensor((1, 1, 32, 64), dtype=DataType.BFLOAT16)
        >>> x.shape
        (1, 1, 32, 64)
        >>> x.bytes()
        4096
        >>> x.logical_volume()
        2048
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: DataType = DataType.BFLOAT16,
        layout: str = "TILE",
        requires_grad: bool = True,
    ):
        """Initialize a MockTensor.

        Args:
            shape: Tensor shape as tuple of ints
            dtype: Data type (default: BFLOAT16)
            layout: Memory layout (default: "TILE")
            requires_grad: Whether gradients should be tracked (default: True)
        """
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.requires_grad = requires_grad

        # Autograd tracking (mirrors ttml.autograd.Tensor)
        self._grad: Optional[MockTensor] = None
        self._node: Optional[BackwardNode] = None

    # =========================================================================
    # ttml.autograd.Tensor interface methods
    # =========================================================================

    def get_value(self) -> "MockTensor":
        """Return self (no wrapped value in MockTensor)."""
        return self

    def get_requires_grad(self) -> bool:
        """Return whether gradients are tracked."""
        return self.requires_grad

    def get_node(self) -> Optional[BackwardNode]:
        """Get the backward node for this tensor."""
        return self._node

    def set_node(self, node: BackwardNode) -> None:
        """Set the backward node for this tensor."""
        self._node = node

    def is_grad_initialized(self) -> bool:
        """Check if gradient has been set."""
        return self._grad is not None

    def get_grad(self) -> Optional["MockTensor"]:
        """Get the gradient tensor."""
        return self._grad

    def add_grad(self, grad: "MockTensor") -> None:
        """Accumulate gradient (for backward pass).

        In a real implementation this would add gradients together.
        For roofline estimation, we just track the latest gradient shape.
        """
        if self._grad is None:
            self._grad = grad
        else:
            # In real impl would add, here just track latest
            self._grad = grad

    # =========================================================================
    # Size/memory calculations
    # =========================================================================

    def logical_volume(self) -> int:
        """Return total number of elements."""
        return math.prod(self.shape)

    def bytes(self) -> int:
        """Return total size in bytes."""
        return int(self.logical_volume() * self.dtype.value)

    # =========================================================================
    # Backward pass
    # =========================================================================

    def backward(self, ctx: "RooflineContext") -> None:
        """Trigger backward pass estimation through the graph.

        Walks backward through the computation graph, calling backward
        functions and accumulating roofline estimates.

        Args:
            ctx: RooflineContext to accumulate estimates into
        """
        # Initialize gradient for output (ones-like)
        if not self.is_grad_initialized():
            self._grad = MockTensor(
                self.shape, self.dtype, self.layout, requires_grad=False
            )

        # Topological sort for backward traversal
        visited = set()
        order: List[MockTensor] = []

        def build_order(tensor: MockTensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))

            if tensor._node is not None:
                for inp in tensor._node.inputs:
                    if inp is not None and inp.requires_grad:
                        build_order(inp)
                order.append(tensor)

        build_order(self)

        # Execute backward in reverse topological order
        for tensor in reversed(order):
            if tensor._node is not None and tensor._node.backward_fn is not None:
                tensor._node.backward_fn(ctx)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def __repr__(self) -> str:
        return f"MockTensor(shape={self.shape}, dtype={self.dtype.name}, requires_grad={self.requires_grad})"

    def clone(self) -> "MockTensor":
        """Create a copy with the same metadata."""
        return MockTensor(
            self.shape,
            self.dtype,
            self.layout,
            self.requires_grad,
        )
