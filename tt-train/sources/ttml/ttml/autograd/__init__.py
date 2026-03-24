# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

    # Re-export C++ classes and enums
    AutoContext = _cpp_autograd.AutoContext
    AutocastTensor = _cpp_autograd.AutocastTensor
    DistributedConfig = _cpp_autograd.DistributedConfig
    GradMode = _cpp_autograd.GradMode
    Graph = _cpp_autograd.Graph
    GraphNode = _cpp_autograd.GraphNode
    NodeId = _cpp_autograd.NodeId
    ParallelismContext = _cpp_autograd.ParallelismContext
    PreferredPrecision = _cpp_autograd.PreferredPrecision
    Tensor = _cpp_autograd.Tensor
    create_tensor = _cpp_autograd.create_tensor

except ImportError:
    # C++ bindings not available yet (e.g., during build)
    pass


def to_numpy(tensor_or_grad, new_type=None, composer=None):
    """Convert a ttml.autograd.Tensor or raw ttnn.Tensor (gradient) to numpy.

    This is a convenience function that handles both:
    - ttml.autograd.Tensor objects (calls tensor.to_numpy())
    - Raw ttnn.Tensor objects (e.g., from tensor.get_grad())

    Args:
        tensor_or_grad: Either a ttml.autograd.Tensor or a raw ttnn.Tensor
        new_type: Optional target data type for conversion (only for autograd.Tensor)
        composer: Optional MeshToTensor composer for distributed tensors

    Returns:
        numpy array with the tensor data
    """
    # Check if it's a ttml.autograd.Tensor by checking for get_value method
    # (both have to_numpy but with different signatures)
    if isinstance(tensor_or_grad, Tensor):
        # It's a ttml.autograd.Tensor - signature: to_numpy(new_type, composer)
        return tensor_or_grad.to_numpy(new_type, composer)

    # It's a raw ttnn.Tensor (e.g., from tensor.get_grad())
    # ttnn.Tensor.to_numpy signature: to_numpy(mesh_composer=None)
    import ttnn

    if isinstance(tensor_or_grad, ttnn.Tensor):
        return tensor_or_grad.to_numpy(composer)

    raise TypeError(f"Cannot convert {type(tensor_or_grad)} to numpy")


__all__ = [
    # Python classes
    "Function",
    "FunctionContext",
    "get_links",
    # Utility functions
    "to_numpy",
    # C++ classes and enums (available after build)
    "AutoContext",
    "AutocastTensor",
    "DistributedConfig",
    "GradMode",
    "Graph",
    "GraphNode",
    "NodeId",
    "ParallelismContext",
    "PreferredPrecision",
    "Tensor",
    "create_tensor",
]
