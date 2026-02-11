# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd function support for TTML.

This module provides a PyTorch-like Function class that allows users to define
custom TTML operations in Python with both forward and backward passes,
integrating with the existing C++ autograd system.
"""

from typing import Any, List, Sequence, Tuple

# Import C++ bindings
from .. import _ttml as cpp


class FunctionContext:
    """Context object for storing tensors and other values needed in backward pass.

    This class is passed to both forward() and backward() methods of a Function.
    Use save_for_backward() in forward() to save tensors, and access them via
    saved_tensors in backward(). You can also set arbitrary attributes on the
    context object (e.g., ctx.scale = 2.0).

    Example:
        def forward(ctx, input, scale):
            ctx.save_for_backward(input)
            ctx.scale = scale  # Store non-tensor value as attribute
            ...

        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            scale = ctx.scale  # Retrieve non-tensor value
            ...
    """

    def __init__(self):
        self._saved_tensors: List[Any] = []

    def save_for_backward(self, *tensors) -> None:
        """Save tensors for use in backward pass.

        Args:
            *tensors: Tensors to save. Can include None values which will be preserved.
        """
        self._saved_tensors = list(tensors)

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve tensors saved in forward pass.

        Returns:
            Tuple of saved tensors in the same order they were saved.
        """
        return tuple(self._saved_tensors)


def _is_ttnn_tensor(obj: Any) -> bool:
    """Check if object is a ttnn tensor (tt::tt_metal::Tensor)."""
    # ttnn tensors have these attributes but not get_requires_grad (which ttml tensors have)
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "get_requires_grad")


def _wrap_output(output: Any) -> Any:
    """Wrap ttnn tensor in ttml autograd tensor if needed."""
    if output is None:
        return None
    if _is_ttnn_tensor(output):
        return cpp.autograd.create_tensor(output, requires_grad=True)
    return output


def _wrap_outputs(outputs: Any) -> Any:
    """Wrap outputs, handling both single tensors and tuples."""
    if isinstance(outputs, tuple):
        return tuple(_wrap_output(out) for out in outputs)
    return _wrap_output(outputs)


def get_links(tensors: Sequence[Any]) -> List[Any]:
    """Extract computation graph links from tensors.

    This function extracts NodeId objects from tensors that have them,
    which are used to connect custom operations into the autograd graph.

    Args:
        tensors: Sequence of tensors (or None values) to extract links from.

    Returns:
        List of NodeId objects from tensors that have nodes set.
    """
    links = []
    for tensor in tensors:
        if tensor is None:
            continue
        node = tensor.get_node()
        if node is not None:
            links.append(node)
    return links


class Function:
    """Base class for defining custom autograd operations.

    Subclass this and implement static forward() and backward() methods
    to create custom operations that integrate with TTML's autograd system.

    There are two usage patterns:

    1. Compose with ttml autograd ops (backward is automatic):
        class MyOp(Function):
            @staticmethod
            def forward(ctx, x, y):
                # Using ttml ops builds the graph automatically
                return x + y  # or ttml.ops.binary.add(x, y)

            # No backward needed - graph already built by autograd ops

    2. Build from ttnn primitives (return gradients like PyTorch):
        class Scale(Function):
            @staticmethod
            def forward(ctx, input, scale_factor):
                ctx.save_for_backward(input)
                ctx.scale_factor = scale_factor
                # Return ttnn tensor - will be auto-wrapped
                return ttnn.multiply(input.get_value(), scale_factor)

            @staticmethod
            def backward(ctx, grad_output):
                # Return gradient for each tensor input (None for non-tensors)
                grad_input = ttnn.multiply(grad_output, ctx.scale_factor)
                return grad_input  # Single tensor input -> single return

        # Use:
        output = Scale.apply(input_tensor, 2.0)
        output.backward()  # Gradients flow to input_tensor
    """

    @staticmethod
    def forward(ctx: FunctionContext, *inputs) -> Any:
        """Compute forward pass of the operation.

        Override this method to implement the forward computation.

        Args:
            ctx: Context object for saving tensors/values for backward.
            *inputs: Input tensors and other arguments.

        Returns:
            Output tensor(s). Can return a single tensor or tuple of tensors.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(ctx: FunctionContext, *grad_outputs) -> Any:
        """Compute backward pass of the operation.

        Override this method when using ttnn primitives in forward().
        Not required when forward() uses ttml autograd ops (graph is automatic).

        Returns gradients in the same order as tensor inputs to forward():
            def backward(ctx, grad_output):
                # Return one gradient per tensor input
                grad_input = ttnn.multiply(grad_output, ctx.scale)
                return grad_input  # or (grad_a, grad_b) for multiple inputs

        Args:
            ctx: Context object with saved tensors/values from forward.
            *grad_outputs: Gradient tensors from upstream (tt::tt_metal::Tensor).
                          One for each output from forward().

        Returns:
            Gradient(s) w.r.t. tensor inputs. Must return same number of
            gradients as there were tensor inputs to forward().
            Use None for inputs that don't need gradients.
        """
        raise NotImplementedError("Subclasses must implement backward()")

    @classmethod
    def apply(cls, *inputs) -> Any:
        """Apply this function to the given inputs.

        This method orchestrates the forward pass and sets up the backward pass
        in the autograd graph.

        Args:
            *inputs: Inputs to pass to forward(). Tensor inputs should come first.

        Returns:
            Output tensor(s) from forward().
        """
        ctx = FunctionContext()

        # Get the AutoContext instance
        auto_context = cpp.autograd.AutoContext.get_instance()

        # Check if gradient mode is enabled
        grad_mode = auto_context.get_gradient_mode()
        grad_enabled = grad_mode == cpp.autograd.GradMode.ENABLED

        # Call forward
        outputs = cls.forward(ctx, *inputs)

        # Auto-wrap ttnn tensors to ttml autograd tensors
        outputs = _wrap_outputs(outputs)

        # If gradients are disabled, just return the outputs
        if not grad_enabled:
            return outputs

        # Normalize outputs to a tuple for consistent handling
        if not isinstance(outputs, tuple):
            outputs_tuple = (outputs,)
        else:
            outputs_tuple = outputs

        # Check if outputs already have nodes from autograd ops
        # If so, the graph is already built and we don't need custom backward
        outputs_have_nodes = all(
            out is not None and out.get_node() is not None for out in outputs_tuple if hasattr(out, "get_node")
        )
        if outputs_have_nodes:
            return outputs

        # Collect input tensors (objects with get_requires_grad method)
        input_tensors = [inp for inp in inputs if hasattr(inp, "get_requires_grad")]

        # Extract links from input tensors (for graph dependencies)
        links = get_links(input_tensors)

        # Create closure factory to properly capture variables
        def make_backward_closure(bwd_ctx, bwd_outputs, bwd_inputs, bwd_cls):
            def backward_closure():
                try:
                    # Get gradients from output tensors
                    # If an output wasn't used in the loss, initialize with zeros
                    # (consistent with C++ behavior where get_grad() returns default tensor)
                    grad_outputs = []
                    for out in bwd_outputs:
                        if out is not None and out.is_grad_initialized():
                            grad_outputs.append(out.get_grad())
                        elif out is not None:
                            # Create zero tensor with same shape as output
                            grad_outputs.append(cpp.core.zeros_like(out.get_value()))
                        else:
                            grad_outputs.append(None)
                    grad_outputs = tuple(grad_outputs)

                    # Call user's backward - returns gradients (PyTorch style)
                    if len(grad_outputs) == 1:
                        grad_inputs = bwd_cls.backward(bwd_ctx, grad_outputs[0])
                    else:
                        grad_inputs = bwd_cls.backward(bwd_ctx, *grad_outputs)

                    # Normalize to tuple
                    if grad_inputs is None:
                        grad_inputs = (None,) * len(bwd_inputs)
                    elif not isinstance(grad_inputs, tuple):
                        grad_inputs = (grad_inputs,)

                    # Validate: number of returned gradients must match number of tensor inputs
                    if len(grad_inputs) != len(bwd_inputs):
                        raise RuntimeError(
                            f"{bwd_cls.__name__}.backward() returned {len(grad_inputs)} "
                            f"gradients but expected {len(bwd_inputs)} (one per tensor input). "
                            f"Return None for inputs that don't need gradients."
                        )

                    # Accumulate gradients to input tensors
                    for tensor, grad in zip(bwd_inputs, grad_inputs):
                        if grad is not None and tensor.get_requires_grad():
                            # Handle both ttml Tensor and raw ttnn tensor
                            if hasattr(grad, "get_value"):
                                tensor.add_grad(grad.get_value())
                            else:
                                tensor.add_grad(grad)
                except Exception as e:
                    raise RuntimeError(f"Error in backward of {bwd_cls.__name__}: {e}") from e

            return backward_closure

        # Create the backward closure with captured variables
        backward_fn = make_backward_closure(ctx, outputs_tuple, input_tensors, cls)

        # Register the backward node
        node_id = auto_context.add_backward_node(backward_fn, links)

        # Set the node on all output tensors
        if node_id is not None:
            if len(outputs_tuple) == 1:
                outputs_tuple[0].set_node(node_id)
            else:
                # Multi-output: Hack, similar to used in multi_head_utils.cpp::grouped_heads_creation
                # Set node on first output only
                outputs_tuple[0].set_node(node_id)

                # Other outputs get dummy nodes that depend on first output
                dummy_links = get_links([outputs_tuple[0]])
                for output in outputs_tuple[1:]:
                    dummy_node = auto_context.add_backward_node(lambda: None, dummy_links)
                    output.set_node(dummy_node)

        return outputs
