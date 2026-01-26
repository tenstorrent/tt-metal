#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simple example: Roofline analysis of 5 different matmul shapes.

This example demonstrates how to use the roofline analysis tool to
estimate performance for matrix multiplications of various sizes.

Run from tt-train directory:
    python3 -m roofline.examples.matmuls
"""

from roofline import (
    MockTensor,
    MockModule,
    MockLinearLayer,
    RooflineContext,
    WORMHOLE_N150,
    DataType,
)


class SimpleLinearModel(MockModule):
    """A simple model with 5 linear layers of different shapes."""

    def __init__(self):
        super().__init__()

        self.layer1 = MockLinearLayer(512, 2048, has_bias=False)
        self.layer2 = MockLinearLayer(2048, 8192, has_bias=False)
        self.layer3 = MockLinearLayer(8192, 2048, has_bias=False)
        self.layer4 = MockLinearLayer(2048, 2048, has_bias=False)
        self.layer5 = MockLinearLayer(2048, 32000, has_bias=False)

    def forward(self, ctx: RooflineContext, x: MockTensor) -> MockTensor:
        x = self.layer1(ctx, x)
        x = self.layer2(ctx, x)
        x = self.layer3(ctx, x)
        x = self.layer4(ctx, x)
        x = self.layer5(ctx, x)
        return x


def main():
    # Create roofline context with Wormhole n150 hardware
    ctx = RooflineContext(WORMHOLE_N150)

    # Create the model
    model = SimpleLinearModel()

    # Input: batch=1, seq_len=1024, features=512
    # Shape: [1, 1, 1024, 512] (4D for ttml compatibility)
    x = MockTensor((1, 1, 1024, 512), dtype=DataType.BFLOAT16, requires_grad=True)

    print("=" * 70)
    print("ROOFLINE ANALYSIS: 5 MatMul Model")
    print("=" * 70)
    print()
    print("Model Architecture:")
    print(f"  Layer 1: 512 -> 2048")
    print(f"  Layer 2: 2048 -> 8192")
    print(f"  Layer 3: 8192 -> 2048")
    print(f"  Layer 4: 2048 -> 2048")
    print(f"  Layer 5: 2048 -> 32000")
    print()
    print(f"Input shape: {x.shape}")
    print(f"Batch size: 1, Sequence length: 1024")
    print()

    # Run forward pass
    print("Running forward pass...")
    y = model(ctx, x)
    print(f"Output shape: {y.shape}")
    print()

    # Run backward pass
    print("Running backward pass...")
    y.backward(ctx)
    print()

    # Print the full summary
    print(ctx.summary(model))

    # Additional analysis: show each layer's contribution
    # Percentages show what fraction of total pass time each layer takes
    print("\nPER-LAYER BREAKDOWN (% of total pass time):")
    print("-" * 70)

    forward_ops = [e for e in ctx.estimates if e.phase == "forward"]
    backward_ops = [e for e in ctx.estimates if e.phase == "backward"]

    total_fwd_time = sum(e.theoretical_time_ns for e in forward_ops)
    total_bwd_time = sum(e.theoretical_time_ns for e in backward_ops)

    print(f"\nForward Pass (total: {total_fwd_time/1e6:.4f} ms):")
    for i, e in enumerate(forward_ops, 1):
        pct = e.theoretical_time_ns / total_fwd_time * 100
        print(
            f"  Layer {i}: {e.theoretical_time_ns/1e6:.4f} ms ({pct:.1f}% of fwd) - {e.bottleneck.value}"
        )

    print(f"\nBackward Pass (total: {total_bwd_time/1e6:.4f} ms):")
    # Backward runs in reverse order: Layer 5 -> Layer 1
    # Each layer has 2 matmuls: grad_input and grad_weight
    layer_names = ["Layer 5", "Layer 4", "Layer 3", "Layer 2", "Layer 1"]
    ops_per_layer = 2  # grad_input and grad_weight (no bias)

    for i, layer_name in enumerate(layer_names):
        start_idx = i * ops_per_layer
        end_idx = start_idx + ops_per_layer
        layer_ops = backward_ops[start_idx:end_idx]
        layer_time = sum(e.theoretical_time_ns for e in layer_ops)
        pct = layer_time / total_bwd_time * 100
        print(f"  {layer_name}: {layer_time/1e6:.4f} ms ({pct:.1f}% of bwd)")


if __name__ == "__main__":
    main()
