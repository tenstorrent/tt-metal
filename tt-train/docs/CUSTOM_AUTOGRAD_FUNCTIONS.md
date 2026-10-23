# Custom Autograd Functions in TTML

This tutorial shows how to define custom operations with automatic differentiation in TTML, following a PyTorch-like API.

## Overview

TTML provides a `Function` base class that lets you define custom operations with forward and backward passes. There are two main patterns:

1. **Compose with existing autograd ops** - backward is automatic
2. **Build from ttnn primitives** - you implement the backward pass

## Basic Usage

```python
import ttml
from ttml.autograd import Function
import ttnn
import numpy as np

# Open device
auto_ctx = ttml.autograd.AutoContext.get_instance()
auto_ctx.open_device()

# Create input tensor
input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
x = ttml.autograd.Tensor.from_numpy(input_data)
```

## Pattern 1: Composing Existing Autograd Operations

When your forward pass uses existing TTML autograd operations (like `+`, `-`, `*`, or `ttml.ops.*`), the computation graph is built automatically and you don't need to implement backward.

```python
class AddAndScale(Function):
    @staticmethod
    def forward(ctx, x, y, scale):
        # Using ttml tensor operations - graph builds automatically
        result = x + y           # Uses ttml tensor __add__
        return result * scale    # Uses ttml tensor __mul__

    # No backward needed - the graph is already built by autograd ops!

# Use it
y = ttml.autograd.Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
output = AddAndScale.apply(x, y, 2.0)

# Backward works automatically
output.backward()
print("x gradient initialized:", x.is_grad_initialized())  # True
```

This pattern is useful when you want to:
- Combine existing operations in a reusable way
- Add custom validation or preprocessing
- Create domain-specific abstractions

## Pattern 2: Building from TTNN Primitives

When you use low-level ttnn operations, you must implement the backward pass yourself. This gives you full control over the gradient computation.

```python
class Scale(Function):
    @staticmethod
    def forward(ctx, input, scale_factor):
        # Save tensors needed for backward
        ctx.save_for_backward(input)
        # Save scalars as attributes
        ctx.scale_factor = scale_factor

        # Use raw ttnn ops - returns ttnn tensor (auto-wrapped to ttml tensor)
        return ttnn.multiply(input.get_value(), scale_factor)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the gradient from upstream (tt::tt_metal::Tensor)

        # Compute gradient: d/dx(x * s) = s
        grad_input = ttnn.multiply(grad_output, ctx.scale_factor)

        # Return gradient for each tensor input
        # (must match number of tensor inputs to forward)
        return grad_input

# Use it
output = Scale.apply(x, 2.0)
output.backward()

# Check gradient: should be 2.0 everywhere
print("x gradient:", x.get_grad())
```

### Key Points for Pattern 2

1. **Save tensors for backward**: Use `ctx.save_for_backward(tensor1, tensor2, ...)` to save tensors you'll need in backward.

2. **Save scalars as attributes**: Use `ctx.scalar_name = value` for non-tensor values.

3. **Return gradients in order**: Backward must return one gradient per tensor input to forward, in the same order. Use `None` for inputs that don't need gradients.

4. **Auto-wrapping**: You can return raw ttnn tensors from forward - they're automatically wrapped as ttml tensors.

## Multiple Inputs and Outputs

### Multiple Tensor Inputs

```python
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return ttnn.add(a.get_value(), b.get_value())

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of a + b is 1 for both inputs
        # Return tuple with one gradient per tensor input
        return grad_output, grad_output

# Use it
a = ttml.autograd.Tensor.from_numpy(np.ones((2, 2), dtype=np.float32))
b = ttml.autograd.Tensor.from_numpy(np.ones((2, 2), dtype=np.float32) * 2)
output = Add.apply(a, b)
output.backward()
```

### Multiple Outputs

```python
class Split(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        value = input.get_value()
        # Return tuple of outputs (both auto-wrapped)
        out1 = ttnn.multiply(value, 2.0)
        out2 = ttnn.multiply(value, 3.0)
        return out1, out2

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        # Receive one gradient per output
        # Combine them for the single input
        grad1 = ttnn.multiply(grad_out1, 2.0)
        grad2 = ttnn.multiply(grad_out2, 3.0)
        return ttnn.add(grad1, grad2)

out1, out2 = Split.apply(x)
```

## Implicit Tensor Conversion

TTML automatically converts between tensor types:

- **Forward output**: ttnn tensors returned from forward are auto-wrapped as ttml autograd tensors
- **Backward input**: grad_output is a raw `tt::tt_metal::Tensor`
- **Backward output**: You can return either raw ttnn tensors or ttml tensors

```python
class Identity(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Return raw ttnn tensor - auto-wrapped
        return input.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is tt::tt_metal::Tensor
        # Can return it directly
        return grad_output
```

## Gradient Validation

TTML validates that backward returns the correct number of gradients:

```python
class BadOp(Function):
    @staticmethod
    def forward(ctx, a, b):  # 2 tensor inputs
        return ttnn.add(a.get_value(), b.get_value())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # ERROR: returns 1 gradient but has 2 inputs!

# This will raise:
# RuntimeError: BadOp.backward() returned 1 gradients but expected 2
```

## Complete Example: Polynomial Function

Similar to [PyTorch's example](https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html), here's a Legendre polynomial P3:

```python
class LegendrePolynomial3(Function):
    """Implements Legendre Polynomial of degree 3: P3(x) = 1/2 * (5x^3 - 3x)"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        x = input.get_value()
        # P3(x) = 0.5 * (5x^3 - 3x)
        x_cubed = ttnn.multiply(ttnn.multiply(x, x), x)  # x^3
        term1 = ttnn.multiply(x_cubed, 5.0)              # 5x^3
        term2 = ttnn.multiply(x, 3.0)                    # 3x
        diff = ttnn.subtract(term1, term2)              # 5x^3 - 3x
        return ttnn.multiply(diff, 0.5)                 # 0.5 * (5x^3 - 3x)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        x = input.get_value()
        # d/dx P3(x) = 0.5 * (15x^2 - 3) = 7.5x^2 - 1.5
        x_squared = ttnn.multiply(x, x)                 # x^2
        term1 = ttnn.multiply(x_squared, 7.5)           # 7.5x^2
        grad_x = ttnn.subtract(term1, 1.5)              # 7.5x^2 - 1.5
        return ttnn.multiply(grad_output, grad_x)

# Use it
x = ttml.autograd.Tensor.from_numpy(np.array([[0.5]], dtype=np.float32))
y = LegendrePolynomial3.apply(x)
y.backward()
```

## Summary

| Pattern | When to Use | Backward Required? |
|---------|-------------|-------------------|
| Compose autograd ops | Combining existing operations | No |
| TTNN primitives | Custom low-level operations | Yes |

Key API:
- `ctx.save_for_backward(tensor1, tensor2, ...)` - Save tensors for backward
- `ctx.attr = value` - Save scalars/other values
- `ctx.saved_tensors` - Retrieve saved tensors in backward
- Return gradients in same order as tensor inputs
- Return `None` for inputs that don't need gradients
