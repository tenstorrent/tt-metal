# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd function support for TTML.

This module provides a PyTorch-like Function class that allows users to define
custom TTML operations in Python with both forward and backward passes,
integrating with the existing C++ autograd system.
"""

from typing import Any, List, Sequence, Tuple

# Import C++ bindings lazily to avoid circular import issues
_cpp_bindings = None


def _get_cpp_bindings():
    """Lazily import C++ bindings to avoid circular import issues."""
    global _cpp_bindings
    if _cpp_bindings is None:
        from ttml import _ttml

        _cpp_bindings = _ttml.autograd
    return _cpp_bindings


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

    The API follows PyTorch's torch.autograd.Function:
    - forward() computes the output and saves tensors/values for backward
    - backward() receives grad_outputs and RETURNS grad_inputs (one per input tensor)

    Example:
        import ttml
        import ttnn

        class Scale(ttml.autograd.Function):
            @staticmethod
            def forward(ctx, input, scale_factor):
                ctx.save_for_backward(input)
                ctx.scale_factor = scale_factor
                output_value = ttnn.multiply(input.get_value(), scale_factor)
                return ttml.autograd.create_tensor(output_value, requires_grad=True)

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                # Return gradient w.r.t. input (None for scale_factor since it's not a tensor)
                grad_value = ttnn.multiply(grad_output, ctx.scale_factor)
                return ttml.autograd.create_tensor(grad_value, requires_grad=False)

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

        Override this method to compute gradients. Unlike the imperative style,
        this method should RETURN the gradients (like PyTorch).

        Args:
            ctx: Context object with saved tensors/values from forward.
            *grad_outputs: Gradient tensors from upstream (tt::tt_metal::Tensor).
                          One for each output from forward().

        Returns:
            Gradients w.r.t. each tensor input from forward(). Return a single
            tensor/value if forward() had one tensor input, or a tuple if multiple.
            Use None for inputs that don't need gradients (non-tensors or tensors
            with requires_grad=False).
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
        cpp = _get_cpp_bindings()
        ctx = FunctionContext()

        # Get the AutoContext instance
        auto_context = cpp.AutoContext.get_instance()

        # Check if gradient mode is enabled
        grad_mode = auto_context.get_gradient_mode()
        grad_enabled = grad_mode == cpp.GradMode.ENABLED

        # Call forward
        outputs = cls.forward(ctx, *inputs)

        # If gradients are disabled, just return the outputs
        if not grad_enabled:
            return outputs

        # Normalize outputs to a tuple for consistent handling
        if not isinstance(outputs, tuple):
            outputs_tuple = (outputs,)
            single_output = True
        else:
            outputs_tuple = outputs
            single_output = False

        # Collect input tensors (objects with get_requires_grad method)
        input_tensors = [inp for inp in inputs if hasattr(inp, "get_requires_grad")]

        # Check if any input requires gradients
        any_requires_grad = any(t.get_requires_grad() for t in input_tensors)

        # Only set up backward if we have inputs that require gradients
        if any_requires_grad:
            # Extract links from input tensors (for graph dependencies)
            links = get_links(input_tensors)

            # Create closure factory to properly capture variables
            def make_backward_closure(bwd_ctx, bwd_outputs, bwd_inputs, bwd_cls):
                def backward_closure():
                    try:
                        # Get gradients from output tensors
                        grad_outputs = []
                        for out in bwd_outputs:
                            if out is not None and out.is_grad_initialized():
                                grad_outputs.append(out.get_grad())
                            else:
                                raise RuntimeError(
                                    f"Output tensor gradient not initialized in "
                                    f"{bwd_cls.__name__}.backward()"
                                )
                        grad_outputs = tuple(grad_outputs)

                        # Call user's backward - returns gradients
                        if len(grad_outputs) == 1:
                            grad_inputs = bwd_cls.backward(bwd_ctx, grad_outputs[0])
                        else:
                            grad_inputs = bwd_cls.backward(bwd_ctx, *grad_outputs)

                        # Normalize to tuple
                        if grad_inputs is None:
                            grad_inputs = (None,) * len(bwd_inputs)
                        elif not isinstance(grad_inputs, tuple):
                            grad_inputs = (grad_inputs,)

                        # Accumulate gradients to input tensors
                        for tensor, grad in zip(bwd_inputs, grad_inputs):
                            if grad is not None and tensor.get_requires_grad():
                                # Handle both Tensor wrapper and raw tt::tt_metal::Tensor
                                if hasattr(grad, "get_value"):
                                    tensor.add_grad(grad.get_value())
                                else:
                                    tensor.add_grad(grad)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error in backward of {bwd_cls.__name__}: {e}"
                        ) from e

                return backward_closure

            # Create the backward closure with captured variables
            backward_fn = make_backward_closure(ctx, outputs_tuple, input_tensors, cls)

            # Register the backward node
            node_id = auto_context.add_backward_node(backward_fn, links)

            # Set the node on all output tensors
            if node_id is not None:
                for output in outputs_tuple:
                    if output is not None and hasattr(output, "set_node"):
                        output.set_node(node_id)

        return outputs if not single_output else outputs
