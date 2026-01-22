# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTML Autograd package.

This package provides autograd functionality for TTML, including:
- C++ autograd classes (Tensor, AutoContext, GradMode, etc.)
- Python custom function support (Function, FunctionContext)

The API follows PyTorch's torch.autograd.Function pattern:

Example usage:
    import ttml
    import ttnn

    # Define custom operation
    class Scale(ttml.autograd.Function):
        @staticmethod
        def forward(ctx, input, scale):
            ctx.save_for_backward(input)
            ctx.scale = scale  # Save scalars as attributes
            output = ttnn.multiply(input.get_value(), scale)
            return ttml.autograd.create_tensor(output, requires_grad=True)

        @staticmethod
        def backward(ctx, grad_output):
            # Return gradient w.r.t. input
            grad = ttnn.multiply(grad_output, ctx.scale)
            return ttml.autograd.create_tensor(grad, requires_grad=False)

    # Use custom operation
    x = ttml.autograd.Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
    y = Scale.apply(x, 2.0)
    y.backward()  # Gradients flow to x
"""

# Import Python implementations first
from .function import Function, FunctionContext, get_links

# Import C++ bindings from _ttml.autograd
try:
    from ttml._ttml import autograd as _cpp_autograd

    # Re-export C++ classes
    AutoContext = _cpp_autograd.AutoContext
    Tensor = _cpp_autograd.Tensor
    GradMode = _cpp_autograd.GradMode
    Graph = _cpp_autograd.Graph
    GraphNode = _cpp_autograd.GraphNode
    AutocastTensor = _cpp_autograd.AutocastTensor
    PreferredPrecision = _cpp_autograd.PreferredPrecision
    create_tensor = _cpp_autograd.create_tensor

except ImportError:
    # C++ bindings not available yet (e.g., during build)
    pass

__all__ = [
    # Python classes
    "Function",
    "FunctionContext",
    "get_links",
    # C++ classes (will be available after build)
    "AutoContext",
    "Tensor",
    "GradMode",
    "Graph",
    "GraphNode",
    "AutocastTensor",
    "PreferredPrecision",
    "create_tensor",
]
