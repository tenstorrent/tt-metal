#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script: Measure actual ttml performance for 5-layer linear model.

This script creates the same model as matmuls.py using actual ttml
LinearLayer modules and measures wall clock time for comparison
with roofline estimates.

Run from tt-train directory:
    python3 -m roofline.examples.benchmark_matmuls
"""

import time
import numpy as np
import ttnn
import ttml
from ttml.autograd import Tensor
from ttml.modules import AbstractModuleBase
from ttml.autograd import Function
import math


def matmul(a, b, transpose_a=False, transpose_b=False):
    device = a.device()
    grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
    return ttnn.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b, core_grid=core_grid
    )


def linear(input, weight, bias=None, transpose_a=False, transpose_b=True):
    device = input.device()
    grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
    return ttnn.linear(
        input,
        weight,
        bias=bias,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        core_grid=core_grid,
    )


class PythonLinearOp(Function):
    """Python implementation of linear operation using ttnn primitives.

    Forward: output = input @ weight.T + bias
    Backward:
        - weight_grad = grad_output.T @ input
        - input_grad = grad_output @ weight
        - bias_grad = sum(grad_output, over batch/sequence dims)
    """

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor | None = None):
        """Forward pass of linear operation.

        Args:
            ctx: Function context for saving tensors
            input: Input tensor of shape [*, in_features]
            weight: Weight tensor of shape [1, 1, out_features, in_features]
            bias: Optional bias tensor of shape [1, 1, 1, out_features]

        Returns:
            Output tensor of shape [*, out_features]
        """
        ctx.save_for_backward(input, weight, bias)

        ttnn_input = input.get_value()
        ttnn_weight = weight.get_value()

        output = linear(
            ttnn_input,
            ttnn_weight,
            bias=bias.get_value() if bias is not None else None,
            transpose_a=False,
            transpose_b=True,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of linear operation.

        Args:
            ctx: Function context with saved tensors
            grad_output: Gradient from upstream, shape [*, out_features]

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        input, weight, bias = ctx.saved_tensors

        ttnn_input = input.get_value()
        ttnn_weight = weight.get_value()

        # Get shapes
        input_shape = ttnn_input.shape
        weight_shape = ttnn_weight.shape

        in_features = input_shape[-1]
        out_features = weight_shape[2]
        volume_without_features = ttnn_input.logical_volume() // in_features

        # Reshape input and grad_output to 2D
        reshaped_input = ttnn.reshape(
            ttnn_input, ttnn.Shape([volume_without_features, in_features])
        )

        reshaped_grad_output = ttnn.reshape(
            grad_output, ttnn.Shape([volume_without_features, out_features])
        )

        # Compute weight gradient: grad_output.T @ input
        # Shape: [out_features, batch*seq] @ [batch*seq, in_features] = [out_features, in_features]
        grad_weight = matmul(
            reshaped_grad_output,
            reshaped_input,
            transpose_a=True,
            transpose_b=False,
        )

        # Reshape to weight's shape [1, 1, out_features, in_features]
        grad_weight = ttnn.reshape(grad_weight, weight_shape)

        # Compute input gradient: grad_output @ weight
        # Shape: [batch*seq, out_features] @ [out_features, in_features] = [batch*seq, in_features]
        grad_input = matmul(
            reshaped_grad_output,
            ttnn_weight,
            transpose_a=False,
            transpose_b=False,  # weight is already [1, 1, out_features, in_features]
        )

        # Reshape back to input's shape
        grad_input = ttnn.reshape(grad_input, input_shape)

        # Compute bias gradient if bias exists
        grad_bias = None
        if bias is not None:
            # Sum over all dimensions except the last (features)
            # reshaped_grad_output is [batch*seq, out_features]
            # We need to sum over dimension 0 to get [out_features]
            grad_bias_flat = ttnn.sum(reshaped_grad_output, dim=0)
            # Reshape to bias shape [1, 1, 1, out_features]
            grad_bias = ttnn.reshape(grad_bias_flat, bias.get_value().shape)

        # TODO: failds if I return grad_bias when has_bias is False
        return grad_input, grad_weight


class PythonLinearLayer(ttml.modules.AbstractModuleBase):
    """Python implementation of Linear layer using PythonLinearOp.

    This is an experimental implementation to test Python operations
    in tt-train's autograd system.
    """

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True
    ) -> None:
        """Initialize Python Linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            has_bias: Whether to include bias term
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        # Initialize weight with uniform distribution [-k, k] where k = sqrt(1/in_features)
        # Weight shape: [1, 1, out_features, in_features]
        init_k = math.sqrt(1.0 / in_features)
        weight_data = np.random.uniform(
            -init_k, init_k, (1, 1, out_features, in_features)
        ).astype(np.float32)
        self.weight = Tensor.from_numpy(weight_data)

        # Initialize bias with same distribution
        # Bias shape: [1, 1, 1, out_features]
        if has_bias:
            bias_data = np.random.uniform(
                -init_k, init_k, (1, 1, 1, out_features)
            ).astype(np.float32)
            self.bias = Tensor.from_numpy(bias_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of linear layer.

        Args:
            x: Input tensor of shape [*, in_features]

        Returns:
            Output tensor of shape [*, out_features]
        """
        return PythonLinearOp.apply(x, self.weight, self.bias)

    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable."""
        return self.forward(x)


class SimpleLinearModel(AbstractModuleBase):
    """A simple model with 5 linear layers of different shapes.

    Same architecture as the roofline example in matmuls.py:
      Layer 1: 512 -> 2048
      Layer 2: 2048 -> 8192
      Layer 3: 8192 -> 2048
      Layer 4: 2048 -> 2048
      Layer 5: 2048 -> 32000
    """

    def __init__(self, layer_dims: list[tuple[int, int]] = None):
        super().__init__()
        dims = layer_dims or LAYER_DIMS
        self.layers = ttml.modules.ModuleList(
            [
                PythonLinearLayer(dims[i][0], dims[i][1], has_bias=False)
                # ttml.modules.LinearLayer(dims[i][0], dims[i][1], has_bias=False)
                for i in range(len(dims))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def calculate_flops(layers: list[tuple[int, int]], M: int = 1024):
    """Calculate total FLOPs for forward + backward pass.

    Args:
        layers: List of (in_features, out_features) tuples for each layer
        M: Number of tokens (batch * seq_len)

    Returns:
        Tuple of (forward_flops, backward_flops, total_flops)
    """
    fwd_flops = 0
    bwd_flops = 0

    for in_f, out_f in layers:
        # Forward: Y = X @ W.T  =>  [M, in_f] @ [in_f, out_f] = 2*M*in_f*out_f
        fwd_flops += 2 * M * in_f * out_f

        # Backward grad_input: dX = dY @ W  =>  [M, out_f] @ [out_f, in_f]
        bwd_flops += 2 * M * out_f * in_f

        # Backward grad_weight: dW = dY.T @ X  =>  [out_f, M] @ [M, in_f]
        bwd_flops += 2 * out_f * M * in_f

    return fwd_flops, bwd_flops, fwd_flops + bwd_flops


# Define layer dimensions in one place for consistency
LAYER_DIMS = [
    (512, 8192),
    (8192, 8192),
    (8192, 8192),
    (8192, 8192),
    (8192, 8192),
]


def benchmark_forward_backward(
    num_warmup: int = 5,
    num_iterations: int = 100,
):
    """Benchmark the 5-layer model forward + backward pass.

    Args:
        num_warmup: Number of warmup iterations for kernel compilation
        num_iterations: Number of timed iterations
    """
    print("=" * 70)
    print("TTML BENCHMARK: 5 MatMul Model")
    print("=" * 70)
    print()
    print("Model Architecture:")
    for i, (in_f, out_f) in enumerate(LAYER_DIMS, 1):
        print(f"  Layer {i}: {in_f} -> {out_f}")
    print()
    print(f"Input shape: (1, 1, 1024, {LAYER_DIMS[0][0]})")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    print()

    # Get device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()
    device = auto_ctx.get_device()

    try:
        # Create model
        model = SimpleLinearModel()

        # Count parameters
        total_params = 0
        for name, param in model.parameters().items():
            param_count = param.get_value().logical_volume()
            total_params += param_count
            print(f"  {name}: {param.get_value().shape}")
        print(
            f"  Total: {total_params:,} params ({total_params * 2 / 1e9:.3f} GB in BF16)"
        )

        # Create input data
        input_shape = (1, 1, 1024, LAYER_DIMS[0][0])
        input_data = np.random.randn(*input_shape).astype(np.float32) * 0.01

        # Warmup phase
        for i in range(num_warmup):
            x = Tensor.from_numpy(input_data.copy())
            y = model(x)
            y.backward(retain_graph=False)
            auto_ctx.reset_graph()

        ttnn.synchronize_device(device)

        # Benchmark phase
        iteration_times = []

        for i in range(num_iterations):
            x = Tensor.from_numpy(input_data.copy())

            iter_start = time.perf_counter()
            y = model(x)
            y.backward(retain_graph=False)
            ttnn.synchronize_device(device)
            iter_end = time.perf_counter()

            iteration_times.append((iter_end - iter_start) * 1000)  # ms
            auto_ctx.reset_graph()

        # Calculate statistics
        avg_time_ms = np.mean(iteration_times)
        std_time_ms = np.std(iteration_times)
        min_time_ms = np.min(iteration_times)
        max_time_ms = np.max(iteration_times)

        # Calculate FLOPs
        M = 1024  # batch * seq_len
        fwd_flops, bwd_flops, total_flops = calculate_flops(LAYER_DIMS, M)
        achieved_tflops = (total_flops / (avg_time_ms / 1000)) / 1e12

        print("Timing:")
        print(f"  Average: {avg_time_ms:.4f} ms")
        print(f"  Std Dev: {std_time_ms:.4f} ms")
        print(f"  Min:     {min_time_ms:.4f} ms")
        print(f"  Max:     {max_time_ms:.4f} ms")
        print()
        print("FLOPs:")
        print(f"  Forward:  {fwd_flops/1e12:.4f} TFLOPs")
        print(f"  Backward: {bwd_flops/1e12:.4f} TFLOPs")
        print(f"  Total:    {total_flops/1e12:.4f} TFLOPs")
        print()
        print(f"Achieved Performance: {achieved_tflops:.2f} TFLOP/s")

        return avg_time_ms, achieved_tflops

    finally:
        auto_ctx.close_device()


def compare_with_roofline():
    """Compare ttml benchmark with roofline estimates."""
    from roofline import (
        MockTensor,
        MockModule,
        MockLinearLayer,
        RooflineContext,
        WORMHOLE_N150,
        DataType,
    )
    from roofline.modules import MockModuleList

    class MockSimpleLinearModel(MockModule):
        def __init__(self, layer_dims: list[tuple[int, int]]):
            super().__init__()
            self.layers = MockModuleList(
                [
                    MockLinearLayer(layer_dims[i][0], layer_dims[i][1], has_bias=False)
                    for i in range(len(layer_dims))
                ]
            )

        def forward(self, ctx: RooflineContext, x: MockTensor) -> MockTensor:
            for layer in self.layers:
                x = layer(ctx, x)
            return x

    # Run roofline estimate
    print()
    print("=" * 70)
    print("ROOFLINE ESTIMATE")
    print("=" * 70)

    ctx = RooflineContext(WORMHOLE_N150)
    model = MockSimpleLinearModel(LAYER_DIMS)
    x = MockTensor(
        (1, 1, 1024, LAYER_DIMS[0][0]), dtype=DataType.BFLOAT16, requires_grad=True
    )

    y = model(ctx, x)
    y.backward(ctx)

    roofline_fwd_ms = ctx.forward_time_ns() / 1e6
    roofline_bwd_ms = ctx.backward_time_ns() / 1e6
    roofline_total_ms = ctx.total_time_ms()
    roofline_tflops = ctx.achieved_tflops()

    print(f"  Forward:  {roofline_fwd_ms:.4f} ms")
    print(f"  Backward: {roofline_bwd_ms:.4f} ms")
    print(f"  Total:    {roofline_total_ms:.4f} ms")
    print(f"  Achieved: {roofline_tflops:.2f} TFLOP/s")

    # Run ttml benchmark
    print()
    ttml_avg_ms, ttml_tflops = benchmark_forward_backward(
        num_warmup=5,
        num_iterations=100,
    )

    # Comparison summary
    print()
    print("=" * 70)
    print("COMPARISON: Roofline vs Actual")
    print("=" * 70)
    print(f"{'Metric':<20} {'Roofline':<15} {'Actual':<15} {'Efficiency':<15}")
    print("-" * 65)
    print(
        f"{'Time (ms)':<20} {roofline_total_ms:<15.4f} {ttml_avg_ms:<15.4f} {roofline_total_ms/ttml_avg_ms*100:<14.1f}%"
    )
    print(
        f"{'TFLOP/s':<20} {roofline_tflops:<15.2f} {ttml_tflops:<15.2f} {ttml_tflops/roofline_tflops*100:<14.1f}%"
    )
    print("=" * 70)
    print()
    print("Note: Efficiency = roofline / actual (100% means hitting roofline)")


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_with_roofline()
    else:
        benchmark_forward_backward(num_warmup=5, num_iterations=100)
        print()
        print("Run with --compare to see roofline comparison")


if __name__ == "__main__":
    main()
