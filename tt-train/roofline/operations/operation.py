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

from ..mock_tensor import MockTensor, BackwardNode

if TYPE_CHECKING:
    from ..roofline import RooflineContext


@dataclass
class RooflineFunctionContext:
    """Context for saving tensors between forward and backward.

    This mirrors ttml.autograd.FunctionContext, allowing operations
    to save inputs needed for backward pass computation.
    """

    _saved_tensors: List[Any] = field(default_factory=list)

    def save_for_backward(self, *tensors: Any) -> None:
        """Save tensors for use in backward pass.

        Args:
            *tensors: Tensors or other values to save
        """
        self._saved_tensors = list(tensors)

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve tensors saved in forward pass."""
        return tuple(self._saved_tensors)


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

        # Track activations for memory estimation
        for out in outputs_tuple:
            if isinstance(out, MockTensor):
                roofline_ctx.track_activation(out)

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

                # Get gradients from output tensors
                grad_outputs = []
                for out in bwd_outputs:
                    if isinstance(out, MockTensor):
                        grad = out.get_grad()
                        if grad is None:
                            grad = MockTensor(out.shape, out.dtype, requires_grad=False)
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
