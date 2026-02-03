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
from enum import Enum
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING
import math

from .hardware import DataType

if TYPE_CHECKING:
    from .roofline import RooflineContext
    from .memory_tracker import MemoryTracker


class TensorLabel(Enum):
    """Label for categorizing tensor memory usage.

    Used for tracking memory allocation by category:
    - PARAMETER: Model weights
    - OPTIMIZER_STATE: Optimizer states (e.g., Adam m and v)
    - ACTIVATION: Forward pass intermediate activations
    - GRADIENT: Backward pass gradients
    """

    PARAMETER = "parameter"
    OPTIMIZER_STATE = "optimizer_state"
    ACTIVATION = "activation"
    GRADIENT = "gradient"


@dataclass
class BackwardNode:
    """Node in the backward computation graph.

    Each node stores a backward function and references to input tensors,
    enabling traversal of the computation graph during backward pass estimation.
    """

    backward_fn: Callable[["RooflineContext"], None]
    inputs: List["MockTensor"] = field(default_factory=list)


# Global memory tracker (set by RooflineContext)
_global_memory_tracker: Optional["MemoryTracker"] = None


def set_global_memory_tracker(tracker: Optional["MemoryTracker"]) -> None:
    """Set the global memory tracker for MockTensor allocations."""
    global _global_memory_tracker
    _global_memory_tracker = tracker


def get_global_memory_tracker() -> Optional["MemoryTracker"]:
    """Get the current global memory tracker."""
    return _global_memory_tracker


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
        label: Optional[TensorLabel] = None,
        name: Optional[str] = None,
    ):
        """Initialize a MockTensor.

        Args:
            shape: Tensor shape as tuple of ints
            dtype: Data type (default: BFLOAT16)
            layout: Memory layout (default: "TILE")
            requires_grad: Whether gradients should be tracked (default: True)
            label: Memory category label (default: None, auto-inferred)
            name: Optional name for tracking (default: None)
        """
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.requires_grad = requires_grad
        self.label = label
        self.name = name

        # Autograd tracking (mirrors ttml.autograd.Tensor)
        self._grad: Optional[MockTensor] = None
        self._node: Optional[BackwardNode] = None

        # Track allocation with global tracker
        self._tracked = False
        if _global_memory_tracker is not None:
            _global_memory_tracker.track_allocation(self)
            self._tracked = True

    def __del__(self):
        """Track deallocation when tensor is garbage collected."""
        if self._tracked and _global_memory_tracker is not None:
            _global_memory_tracker.track_deallocation(self)

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

    def backward(self, ctx: "RooflineContext", retain_graph: bool = False) -> None:
        """Trigger backward pass estimation through the graph.

        Walks backward through the computation graph, calling backward
        functions and accumulating roofline estimates.

        When retain_graph=False (default), activations and gradients are
        deallocated as soon as possible during backward pass, reducing
        peak memory usage. This matches ttml behavior.

        Args:
            ctx: RooflineContext to accumulate estimates into
            retain_graph: If False, release memory as we go (default).
                         If True, keep graph for multiple backward passes.
        """
        import gc

        # Initialize gradient for output (ones-like)
        if not self.is_grad_initialized():
            self._grad = MockTensor(
                self.shape,
                self.dtype,
                self.layout,
                requires_grad=False,
                label=TensorLabel.GRADIENT,
                name="loss_grad",
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
        # Pop from list to release references as we go
        while order:
            tensor = order.pop()  # Pop from end (reverse order)
            if tensor._node is not None and tensor._node.backward_fn is not None:
                tensor._node.backward_fn(ctx)

                # Release memory if not retaining graph (like ttml)
                if not retain_graph:
                    # Clear saved tensors to release activation memory
                    # The backward closure holds a reference to RooflineFunctionContext
                    # which holds saved_tensors. Clearing backward_fn releases these.
                    tensor._node.backward_fn = None
                    tensor._node.inputs = []

                    # Clear gradient to release gradient memory
                    # (downstream nodes have already used it)
                    tensor._grad = None

                    # Clear the node itself to break any remaining references
                    tensor._node = None

                    # Force GC after each node to trigger deallocations
                    # This is needed because Python doesn't immediately collect
                    gc.collect()

        # Final GC pass to clean up any remaining objects
        if not retain_graph:
            gc.collect()

    # =========================================================================
    # Utility methods
    # =========================================================================

    def __repr__(self) -> str:
        label_str = f", label={self.label.value}" if self.label else ""
        name_str = f", name={self.name}" if self.name else ""
        return f"MockTensor(shape={self.shape}, dtype={self.dtype.name}, requires_grad={self.requires_grad}{label_str}{name_str})"

    def clone(self) -> "MockTensor":
        """Create a copy with the same metadata."""
        return MockTensor(
            self.shape,
            self.dtype,
            self.layout,
            self.requires_grad,
            self.label,
            self.name,
        )
