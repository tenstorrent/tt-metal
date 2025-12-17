#!/usr/bin/env python3
"""
Example script showing how to use the convenience functions
to get validated input tensors from generated reference files.
"""

import torch
from load_torch_reference_outputs import (
    get_validated_inputs,
    get_conv2d_inputs,
    get_matmul_inputs,
    get_reference_outputs,
)

print("=" * 80)
print("Example: Using Reference Data in Your Script")
print("=" * 80)

# ==============================================================================
# Example 1: Get conv2d inputs for testing
# ==============================================================================
print("\n[Example 1] Get Conv2D inputs for testing:")
print("-" * 80)

# Get validated conv2d inputs
conv_input, conv_weight, conv_bias = get_conv2d_inputs(dtype="bfloat16", input_channels=8, method="rand")

print(f"Input:  {conv_input.shape} {conv_input.dtype}")
print(f"Weight: {conv_weight.shape} {conv_weight.dtype}")
print(f"Bias:   {conv_bias.shape} {conv_bias.dtype}")

# Now you can use these tensors for your tests
# For example, run through TTNN or PyTorch
print("\n# You can now use these tensors in your code:")
print("# ttnn_output = ttnn.conv2d(conv_input, conv_weight, conv_bias, ...)")
print("# torch_output = torch.nn.functional.conv2d(conv_input, conv_weight, conv_bias, ...)")

# ==============================================================================
# Example 2: Get matmul inputs for testing
# ==============================================================================
print("\n[Example 2] Get Matmul inputs for testing:")
print("-" * 80)

A, B = get_matmul_inputs(dtype="bfloat16", input_channels=256, method="randn")

print(f"A: {A.shape} {A.dtype}")
print(f"B: {B.shape} {B.dtype}")

print("\n# You can now use these tensors:")
print("# ttnn_result = ttnn.matmul(A, B)")
print("# torch_result = torch.matmul(A, B)")

# ==============================================================================
# Example 3: Get everything at once
# ==============================================================================
print("\n[Example 3] Get all inputs at once:")
print("-" * 80)

all_inputs = get_validated_inputs(dtype="float32", input_channels=64, method="rand")

print("Conv2D inputs:")
print(f"  input:  {all_inputs['conv2d']['input'].shape}")
print(f"  weight: {all_inputs['conv2d']['weight'].shape}")
print(f"  bias:   {all_inputs['conv2d']['bias'].shape}")

print("\nMatmul inputs:")
print(f"  A: {all_inputs['matmul']['A'].shape}")
print(f"  B: {all_inputs['matmul']['B'].shape}")

print("\nVerification status:")
print(f"  All passed: {all_inputs['verification']['all_passed']}")

# ==============================================================================
# Example 4: Get reference outputs for comparison
# ==============================================================================
print("\n[Example 4] Get reference outputs for comparison:")
print("-" * 80)

reference = get_reference_outputs(dtype="bfloat16", input_channels=8, method="rand")

print(f"Conv2D reference output: {reference['conv2d_output'].shape}")
print(f"Matmul reference output: {reference['matmul_output'].shape}")

print("\n# Compare your output against reference:")
print("# pcc = compute_pcc(your_output, reference['conv2d_output'])")
print("# ulp = compute_ulp_error(your_output, reference['conv2d_output'])")

# ==============================================================================
# Example 5: Loop over multiple test cases
# ==============================================================================
print("\n[Example 5] Loop over multiple test cases:")
print("-" * 80)

test_configs = [
    ("bfloat16", 8, "rand"),
    ("bfloat16", 16, "rand"),
    ("bfloat16", 32, "randn"),
]

for dtype, ic, method in test_configs:
    inputs = get_conv2d_inputs(dtype, ic, method)
    print(f"  {dtype:9s} | ic={ic:3d} | {method:5s} | input={inputs[0].shape}")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print("=" * 80)
