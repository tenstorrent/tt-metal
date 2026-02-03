# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RooflineFunction base class for roofline-aware operations.

This module provides the RooflineFunction and RooflineFunctionContext
base classes for defining roofline-aware operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor, BackwardNode, TensorLabel
from ..hardware import DataType

if TYPE_CHECKING:
    from ..roofline import RooflineContext


def create_grad_tensor(
    shape: Tuple[int, ...],
    dtype: DataType = DataType.BFLOAT16,
    layout: str = "TILE",
    name: Optional[str] = None,
) -> MockTensor:
    """Create a gradient tensor with proper GRADIENT label.

    This is a helper function for backward methods to create properly
    labeled gradient tensors.

    Args:
        shape: Tensor shape
        dtype: Data type (default: BFLOAT16)
        layout: Memory layout (default: "TILE")
        name: Optional name for tracking

    Returns:
        MockTensor with GRADIENT label
    """
    return MockTensor(
        shape,
        dtype,
        layout,
        requires_grad=False,
        label=TensorLabel.GRADIENT,
        name=name,
    )


def create_activation_tensor(
    shape: Tuple[int, ...],
    dtype: DataType = DataType.BFLOAT16,
    layout: str = "TILE",
    requires_grad: bool = True,
    name: Optional[str] = None,
) -> MockTensor:
    """Create an activation tensor with proper ACTIVATION label.

    This is a helper function for forward methods to create properly
    labeled output tensors that will be saved for backward pass.

    Args:
        shape: Tensor shape
        dtype: Data type (default: BFLOAT16)
        layout: Memory layout (default: "TILE")
        requires_grad: Whether gradients should be tracked (default: True)
        name: Optional name for tracking

    Returns:
        MockTensor with ACTIVATION label
    """
    return MockTensor(
        shape,
        dtype,
        layout,
        requires_grad=requires_grad,
        label=TensorLabel.ACTIVATION,
        name=name,
    )


@dataclass
class RooflineFunctionContext:
    """Context for saving tensors between forward and backward.

    This mirrors ttml.autograd.FunctionContext, allowing operations
    to save inputs needed for backward pass computation.

    Tensors saved here represent activations that must be kept in memory
    until the backward pass completes.
    """

    _saved_tensors: List[Any] = field(default_factory=list)

    def save_for_backward(self, *tensors: Any) -> None:
        """Save tensors for use in backward pass.

        These tensors represent activations that must be kept in memory
        until backward pass completes.

        Note: Tensors should be created with proper labels using
        create_activation_tensor() for activations or MockParameter for parameters.

        Args:
            *tensors: Tensors or other values to save
        """
        from ..mock_tensor import TensorLabel

        # Verify MockTensors have proper labels
        for t in tensors:
            if isinstance(t, MockTensor):
                assert t.label in (
                    TensorLabel.ACTIVATION,
                    TensorLabel.PARAMETER,
                    None,
                ), (
                    f"Tensor saved for backward must be labeled as ACTIVATION or PARAMETER, "
                    f"got {t.label}. Use create_activation_tensor() for forward outputs "
                    f"or MockParameter for parameters."
                )

        self._saved_tensors = list(tensors)

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve tensors saved in forward pass."""
        return tuple(self._saved_tensors)

    def clear_saved_tensors(self) -> None:
        """Clear saved tensors to release activation memory.

        Called after backward pass when retain_graph=False to allow
        early deallocation of activations.
        """
        self._saved_tensors.clear()


class RooflineFunction:
    """Base class for roofline-aware operations.

    Mirrors ttml.autograd.Function but for roofline estimation:
    - forward() receives MockTensors and RooflineContext
    - backward() estimates backward pass performance
    - apply() orchestrates forward/backward and builds graph

    Subclasses should implement forward() and backward() as static methods.

    Example:
        >>> class MyOp(RooflineFunction):
        ...     @staticmethod
        ...     def forward(ctx, roofline_ctx, input, weight):
        ...         ctx.save_for_backward(input, weight)
        ...         # Add forward estimate to roofline_ctx
        ...         # Return output MockTensor
        ...         pass
        ...
        ...     @staticmethod
        ...     def backward(ctx, roofline_ctx, grad_output):
        ...         # Add backward estimates to roofline_ctx
        ...         # Return gradient MockTensors
        ...         pass
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        *inputs: Any,
    ) -> MockTensor:
        """Compute forward pass and add estimate to context.

        Args:
            ctx: Function context for saving tensors
            roofline_ctx: Roofline context for accumulating estimates
            *inputs: Input MockTensors and other arguments

        Returns:
            Output MockTensor(s)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[Optional[MockTensor], ...]:
        """Compute backward pass estimates.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for accumulating estimates
            grad_output: Gradient tensor from upstream

        Returns:
            Tuple of gradient MockTensors (one per tensor input)
        """
        raise NotImplementedError("Subclasses must implement backward()")

    @classmethod
    def apply(
        cls,
        roofline_ctx: "RooflineContext",
        *inputs: Any,
    ) -> MockTensor:
        """Apply this function to the given inputs.

        This method orchestrates the forward pass and sets up the
        backward pass in the computation graph.

        Args:
            roofline_ctx: Roofline context for estimates
            *inputs: Inputs to pass to forward()

        Returns:
            Output MockTensor(s) from forward()
        """
        ctx = RooflineFunctionContext()

        # Call forward
        outputs = cls.forward(ctx, roofline_ctx, *inputs)

        # Normalize outputs to tuple for consistent handling
        if not isinstance(outputs, tuple):
            outputs_tuple = (outputs,)
        else:
            outputs_tuple = outputs

        # Note: Activations are now labeled in save_for_backward()
        # Output tensors that aren't saved for backward will remain unlabeled
        # (they're intermediate and will be deallocated)

        # Collect input tensors that need gradients
        input_tensors = [
            inp for inp in inputs if isinstance(inp, MockTensor) and inp.requires_grad
        ]

        # Create backward closure factory
        def make_backward_closure(
            bwd_ctx, bwd_roofline_ctx, bwd_outputs, bwd_inputs, bwd_cls
        ):
            def backward_fn(ctx_for_bwd: "RooflineContext"):
                # Import here to avoid circular dependency
                from ..roofline import elementwise_roofline
                from ..mock_tensor import TensorLabel

                # Get gradients from output tensors
                grad_outputs = []
                for out in bwd_outputs:
                    if isinstance(out, MockTensor):
                        grad = out.get_grad()
                        if grad is None:
                            grad = MockTensor(
                                out.shape,
                                out.dtype,
                                requires_grad=False,
                                label=TensorLabel.GRADIENT,
                                name=f"grad_{out.name}" if out.name else None,
                            )
                        grad_outputs.append(grad)
                    else:
                        grad_outputs.append(None)
                grad_outputs = tuple(grad_outputs)

                # Call backward - single grad for single output, multiple for multi-output
                if len(grad_outputs) == 1:
                    grad_inputs = bwd_cls.backward(
                        bwd_ctx, bwd_roofline_ctx, grad_outputs[0]
                    )
                else:
                    grad_inputs = bwd_cls.backward(
                        bwd_ctx, bwd_roofline_ctx, *grad_outputs
                    )

                # Normalize to tuple
                if grad_inputs is None:
                    grad_inputs = (None,) * len(bwd_inputs)
                elif not isinstance(grad_inputs, tuple):
                    grad_inputs = (grad_inputs,)

                # Accumulate gradients to input tensors
                for tensor, grad in zip(bwd_inputs, grad_inputs):
                    if grad is not None and tensor.requires_grad:
                        # If tensor already has a gradient, add roofline estimate for accumulation
                        if tensor.is_grad_initialized():
                            num_elements = tensor.logical_volume()
                            estimate = elementwise_roofline(
                                bwd_roofline_ctx.hw,
                                num_elements,
                                num_inputs=2,  # read existing grad + new grad
                                sfpu_ops_per_element=0.0,
                                fpu_ops_per_element=1.0,  # addition
                                dtype=tensor.dtype,
                                operation="GradAccum",
                                phase="backward",
                            )
                            bwd_roofline_ctx.add_perf_result(estimate)
                        tensor.add_grad(grad)

                # Clear saved tensors to allow early deallocation of activations
                # This happens after backward computes gradients but before
                # MockTensor.backward() clears the entire backward_fn
                bwd_ctx.clear_saved_tensors()

            return backward_fn

        # Create backward closure
        backward_fn = make_backward_closure(
            ctx, roofline_ctx, outputs_tuple, input_tensors, cls
        )

        # Set node on outputs (similar to ttml's multi-output handling)
        if len(outputs_tuple) == 1:
            outputs_tuple[0].set_node(BackwardNode(backward_fn, input_tensors))
        else:
            # Multi-output: set main backward node on first output only
            outputs_tuple[0].set_node(BackwardNode(backward_fn, input_tensors))

            # Other outputs get dummy nodes that depend on first output
            # This ensures proper backward traversal order
            for out in outputs_tuple[1:]:
                if isinstance(out, MockTensor):
                    out.set_node(BackwardNode(lambda ctx: None, [outputs_tuple[0]]))

        return outputs
