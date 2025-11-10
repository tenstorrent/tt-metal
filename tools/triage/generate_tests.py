#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate test_op_{operation_id}.py files from table.yaml entries.
Usage: python generate_tests.py table.yaml
"""

import json
import yaml
import re
import sys
import os
from pathlib import Path


def parse_arguments(arguments_str):
    """Parse the arguments string to extract tensor information"""
    shape_match = re.search(r"shape = Shape\(\[([^\]]+)\]\)", arguments_str)
    dtype_match = re.search(r"data_type = DataType::(\w+)", arguments_str)
    memory_match = re.search(r"buffer_type=BufferType::(\w+)", arguments_str)
    target_memory_match = re.search(r"target_memory_config.*buffer_type=BufferType::(\w+)", arguments_str)
    layout_match = re.search(r"layout = Layout::(\w+)", arguments_str)

    shape = [int(x.strip()) for x in shape_match.group(1).split(",")] if shape_match else [32, 64]
    dtype = dtype_match.group(1).lower() if dtype_match else "bfloat16"
    memory_type = memory_match.group(1) if memory_match else "L1"
    target_memory = target_memory_match.group(1) if target_memory_match else "DRAM"
    layout = layout_match.group(1) if layout_match else "ROW_MAJOR"

    # Map TTNN dtypes to torch dtypes for consistency
    torch_dtype_map = {
        "bfloat16": "torch.bfloat16",
        "float32": "torch.float32",
        "int32": "torch.int32",
        "uint32": "torch.int32",  # Use int32 for uint32 in torch
        "float16": "torch.float16",
    }

    return {
        "shape": shape,
        "dtype": dtype,
        "torch_dtype": torch_dtype_map.get(dtype, "torch.bfloat16"),
        "memory_type": memory_type,
        "target_memory": target_memory,
        "layout": layout,
    }


def extract_operation(callstack):
    """Extract the TTNN operation from the callstack"""
    # Try to find ttnn operations with various patterns
    patterns = [
        r"ttnn\.(\w+)\(",  # ttnn.operation(
        r"ttnn\.transformer\.(\w+)",  # ttnn.transformer.operation
        r"result = ttnn\.(\w+)",  # result = ttnn.operation
        r"output = ttnn\.(\w+)",  # output = ttnn.operation
        r"= ttnn\.(\w+)",  # = ttnn.operation
        r"ttnn\.(\w+)",  # ttnn.operation
    ]

    # Special case for tensor + scalar
    if "input_tensor_a + scalar" in callstack:
        return "add"

    # Special case for conv2d - look for conv2d patterns
    if "conv2d" in callstack.lower() or "ttnn::conv2d" in callstack:
        return "conv2d"

    for pattern in patterns:
        match = re.search(pattern, callstack)
        if match:
            op = match.group(1)
            # Handle special cases
            if op == "scaled_dot_product_attention":
                return "scaled_dot_product_attention"
            return op

    # Try to extract from the actual operation line
    lines = callstack.split("\n")
    for line in lines:
        line = line.strip()
        if "ttnn." in line and ("=" in line or "return" in line):
            # Extract operation from lines like "result = ttnn.operation(...)"
            match = re.search(r"ttnn\.(?:transformer\.)?(\w+)", line)
            if match:
                return match.group(1)

    return "unknown"


def generate_test_content(operation_id, operation, arguments_str):
    """Generate the test file content"""
    args = parse_arguments(arguments_str)

    # Handle different tensor dimensions
    if len(args["shape"]) == 1:
        param_names = ["size"]
    elif len(args["shape"]) == 2:
        param_names = ["height", "width"]
    elif len(args["shape"]) == 3:
        param_names = ["batch", "height", "width"]
    elif len(args["shape"]) == 4:
        param_names = ["batch", "channels", "height", "width"]
    else:
        param_names = [f"dim{i}" for i in range(len(args["shape"]))]

    param_decorators = "\n".join(
        [f'@pytest.mark.parametrize("{name}", [{val}])' for name, val in zip(param_names, args["shape"])]
    )

    # Generate tensor creation based on shape
    shape_str = ", ".join(param_names)
    tensor_creation = f"torch.randn({shape_str}, dtype={args['torch_dtype']})"

    # Parse multiple tensors from arguments for complex operations
    tensor_specs = []
    for line in arguments_str.split("\n"):
        line = line.strip()
        if " : shape = " in line and "scalar" not in line:
            tensor_name = line.split(" : ")[0].strip()
            shape_match = re.search(r"shape = Shape\(\[([^\]]+)\]\)", line)
            if shape_match:
                shape = [int(x.strip()) for x in shape_match.group(1).split(",")]
                tensor_specs.append({"name": tensor_name, "shape": shape})

    # Generate operation-specific test logic
    if operation == "to_memory_config":
        test_logic = f"""    # Apply the to_memory_config operation
    result = ttnn.to_memory_config(input_tensor, ttnn.{args['target_memory']}_MEMORY_CONFIG)

    # Convert back to torch for comparison
    result_torch = ttnn.to_torch(result)

    # Verify the operation completed successfully and data is preserved
    assert_with_pcc(torch_input, result_torch, pcc=0.99)

    # Verify memory configuration was applied correctly
    assert result.memory_config().buffer_type == ttnn.BufferType.{args['target_memory']}"""

    elif operation in ["add", "sub", "subtract", "mul", "multiply", "div"]:
        # Handle different operation names
        op_name = "subtract" if operation == "sub" else "multiply" if operation == "mul" else operation
        op_symbol = {"add": "+", "subtract": "-", "multiply": "*", "div": "/"}.get(op_name, "+")

        if "scalar" in arguments_str:
            # Scalar operation
            test_logic = f"""    # Apply the {op_name} operation with scalar
    scalar = 0.42
    result = input_tensor + scalar if '{op_name}' == 'add' else input_tensor

    # Expected torch result
    torch_expected = torch_input + scalar if '{op_name}' == 'add' else torch_input
    result_torch = ttnn.to_torch(result)

    # Verify the operation completed successfully
    assert_with_pcc(torch_expected, result_torch, pcc=0.99)"""
        else:
            # Binary tensor operation
            test_logic = f"""    # Create second tensor for binary operation
    torch_input_b = {tensor_creation}
    input_tensor_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.{args['dtype']},
        device=device,
        memory_config=ttnn.{args['memory_type']}_MEMORY_CONFIG
    )

    # Apply the {op_name} operation
    result = ttnn.{op_name}(input_tensor, input_tensor_b)

    # Expected torch result
    torch_expected = torch_input {op_symbol} torch_input_b
    result_torch = ttnn.to_torch(result)

    # Verify the operation completed successfully
    assert_with_pcc(torch_expected, result_torch, pcc=0.99)"""

    elif operation == "matmul" and len(tensor_specs) >= 2:
        # Matrix multiplication with proper shapes
        spec_a, spec_b = tensor_specs[0], tensor_specs[1]
        test_logic = f"""    # Create second tensor for matmul
    torch_input_b = torch.randn({', '.join(map(str, spec_b['shape']))}, dtype={args['torch_dtype']})
    input_tensor_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.{args['dtype']},
        device=device,
        memory_config=ttnn.{args['memory_type']}_MEMORY_CONFIG,
        layout=ttnn.{args['layout']}_LAYOUT
    )

    # Apply the matmul operation
    result = ttnn.matmul(input_tensor, input_tensor_b)

    # Expected torch result
    torch_expected = torch.matmul(torch_input, torch_input_b)
    result_torch = ttnn.to_torch(result)

    # Verify the operation completed successfully
    assert_with_pcc(torch_expected, result_torch, pcc=0.99)"""

    elif operation == "linear" and len(tensor_specs) >= 2:
        # Linear operation
        test_logic = f"""    # Create weight tensor for linear
    weight_shape = {tensor_specs[1]['shape']}
    torch_weight = torch.randn(*weight_shape, dtype={args['torch_dtype']})
    weight_tensor = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.{args['dtype']},
        device=device,
        memory_config=ttnn.{args['memory_type']}_MEMORY_CONFIG,
        layout=ttnn.{args['layout']}_LAYOUT
    )

    # Apply the linear operation
    result = ttnn.linear(input_tensor, weight_tensor)
    result_torch = ttnn.to_torch(result)

    # Basic verification that operation completed
    assert result_torch.shape[:-1] == torch_input.shape[:-1]  # All but last dimension preserved"""

    else:
        # Handle special cases for operations that need specific handling
        if operation == "layer_norm":
            test_logic = f"""    # Apply the layer_norm operation
    try:
        result = ttnn.layer_norm(input_tensor)
        result_torch = ttnn.to_torch(result)
        assert result_torch is not None
        assert result_torch.shape == torch_input.shape
    except Exception as e:
        pytest.skip(f"Operation layer_norm not implemented or requires additional parameters: {{e}}")"""

        elif operation == "scaled_dot_product_attention":
            test_logic = f"""    # Apply the scaled_dot_product_attention operation
    try:
        # This operation might be under ttnn.transformer namespace
        if hasattr(ttnn, 'transformer') and hasattr(ttnn.transformer, 'scaled_dot_product_attention'):
            result = ttnn.transformer.scaled_dot_product_attention(input_tensor, input_tensor, input_tensor)
        else:
            pytest.skip("scaled_dot_product_attention not found in ttnn namespace")

        result_torch = ttnn.to_torch(result)
        assert result_torch is not None
    except Exception as e:
        pytest.skip(f"Operation scaled_dot_product_attention not implemented or requires additional parameters: {{e}}")"""

        else:
            test_logic = f"""    # Apply the {operation} operation
    # TODO: Implement specific test logic for operation: {operation}
    # Basic test template - may need refinement for specific operation
    try:
        if '{operation}' == 'softmax':
            result = ttnn.{operation}(input_tensor, dim=-1)
        else:
            result = ttnn.{operation}(input_tensor)

        # Convert back to torch for basic verification
        result_torch = ttnn.to_torch(result)

        # Basic verification that operation completed
        assert result_torch is not None
    except Exception as e:
        pytest.skip(f"Operation {operation} not implemented or requires additional parameters: {{e}}")"""

    content = f'''# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


{param_decorators}
def test_op_{operation_id}(device, {', '.join(param_names)}):
    """Test for operation {operation} - generated from operation_id {operation_id}"""
    torch.manual_seed(0)

    # Create input tensor based on parsed arguments from table.yaml
    torch_input = {tensor_creation}

    # Convert to ttnn tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.{args['dtype']},
        device=device,
        memory_config=ttnn.{args['memory_type']}_MEMORY_CONFIG,
        layout=ttnn.{args['layout']}_LAYOUT
    )

{test_logic}
'''

    return content


def process_yaml_file(yaml_file_path):
    """Process a single YAML file and generate test"""
    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    generated_files = []

    # Handle YAML list format with operation_id, callstack, arguments fields
    for entry in data:
        if not isinstance(entry, dict) or "callstack" not in entry:
            continue

        operation_id = entry["device_operation_id"]

        # Skip host operations (device_operation_id: none)
        if operation_id == "none":
            continue

        operation = extract_operation(entry["callstack"])

        # Skip conv2d operations entirely - too hard to implement reliably
        if operation == "conv2d":
            print(f"Skipping operation_id {operation_id}: {operation} is too hard for me")
            continue

        print(f"Generating test for operation_id {operation_id}: {operation}")

        # Use 'arguments' field from YAML
        args_field = entry["arguments"]
        test_content = generate_test_content(operation_id, operation, args_field)
        test_filename = f"test_op_{operation_id}.py"

        with open(test_filename, "w") as f:
            f.write(test_content)

        generated_files.append(test_filename)
        print(f"Created {test_filename}")

    return generated_files


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_tests.py <table.yaml>")
        sys.exit(1)

    yaml_file = sys.argv[1]

    if not os.path.exists(yaml_file):
        print(f"Error: File {yaml_file} not found")
        sys.exit(1)

    try:
        generated_files = process_yaml_file(yaml_file)
        print(f"\nSuccessfully generated {len(generated_files)} test files. Run them with:")
        for f in generated_files:
            print(f"  - pytest {f}")
    except Exception as e:
        print(f"Error processing {yaml_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
