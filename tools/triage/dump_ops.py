#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_ops [--generate-test] [--max-width=<width>]

Options:
    --generate-test        Generate test files for hanging operations
    --max-width=<width>    Maximum column width for table output. [default: 100]

Description:
    Prints the current operation running on each core.

    Use --generate-test to create test files for operations that could be causing hangs.
    Use --max-width to control table column widths.

"""

from triage import ScriptConfig, triage_field, run_script
from dataclasses import dataclass
from run_checks import run as get_run_checks
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from inspector_data import run as get_inspector_data

try:
    from ttexalens.context import Context
    from ttexalens.device import Device
except ImportError:
    print("Error: ttexalens module not found.")
    print("Please run 'scripts/install_debugger.sh' to install the required debugging dependencies.")
    exit(1)

import re, textwrap, subprocess, shutil
import os
from pathlib import Path

script_config = ScriptConfig(
    data_provider=False,
    depends=["inspector_data", "dispatcher_data"],
)


# ============================================================================
# Test Generation Functions (from generate_tests.py)
# ============================================================================


def parse_arguments_for_test(arguments_str):
    """Parse the arguments string to extract tensor information for test generation"""
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


def extract_operation_from_name(operation_name):
    """Extract the operation from ttnn operation name (e.g., ttnn::add -> add)"""
    # Remove ttnn:: prefix if present
    if operation_name.startswith("ttnn::"):
        return operation_name[6:]
    elif operation_name.startswith("ttnn."):
        return operation_name[5:]
    return operation_name


def generate_test_content(operation_id, operation, arguments_str):
    """Generate the test file content for a specific operation"""
    args = parse_arguments_for_test(arguments_str)

    # Create parameter strings for pytest
    shape_params = ", ".join(str(s) for s in args["shape"])

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

    else:
        # Generic test logic for other operations
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

    # Create input tensor based on parsed arguments
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


def generate_tests_for_operations(host_id_op_names, inspector_data):
    """Generate test files for operations that might be causing hangs"""

    # Get operations mapping from Inspector
    host_id_mapping = fetch_operations_from_inspector(inspector_data) if inspector_data else {}

    if not host_id_mapping:
        print("[Warning] No operations data available from Inspector. Cannot generate tests.")
        return

    generated_files = []

    print("\n" + "=" * 60)
    print("Generating test files for hanging operations...")
    print("=" * 60)

    # Process each unique operation
    seen_operations = set()

    for host_id, op_name in host_id_op_names:
        operation_id_key = str(host_id)

        # Skip if we've already generated a test for this operation
        if operation_id_key in seen_operations:
            continue

        # Skip host-only operations
        if operation_id_key not in host_id_mapping:
            continue

        mapping = host_id_mapping[operation_id_key]
        if mapping.get("device_operation_id") == "none":
            continue

        seen_operations.add(operation_id_key)

        operation_name = mapping.get("operation_name", "unknown_op")
        callstack = mapping.get("callstack", "")
        arguments = mapping.get("arguments", "")

        # Extract clean operation name (e.g., ttnn::add -> add)
        operation = extract_operation_from_name(operation_name)

        # Skip conv2d operations - too complex
        if operation == "conv2d":
            print(f"Skipping operation_id {host_id}: {operation} (too complex for automated test generation)")
            continue

        print(f"\nGenerating test for operation_id {host_id}: {operation_name}")

        # Show callstack to help identify the hanging operation
        if callstack:
            print(f"  Location: {callstack}")

        try:
            # Generate test content
            test_content = generate_test_content(host_id, operation, arguments)
            test_filename = f"test_op_{host_id}.py"

            # Write test file
            with open(test_filename, "w") as f:
                f.write(test_content)

            generated_files.append(test_filename)
            print(f"  ✓ Created {test_filename}")
            print(f"     Run with: pytest {test_filename} -xvs")

        except Exception as e:
            print(f"  ✗ Failed to generate test: {e}")

    # Print summary
    print("\n" + "=" * 60)
    if generated_files:
        print(f"Successfully generated {len(generated_files)} test files to isolate the hanging operation:")
        print("")
        print("Run all tests together:")
        print(f"  pytest {' '.join(generated_files)} -xvs")
        print("")
        print("Or run individually:")
        for f in generated_files:
            print(f"  pytest {f}")
        print("")
        print("The test that hangs will identify which operation is causing the issue.")
    else:
        print("No test files were generated.")
    print("=" * 60)


# Color constants for argument highlighting
RST = "\033[0m"
COLORS = [
    "\033[31m",  # Red
    "\033[32m",  # Green
    "\033[33m",  # Yellow
    "\033[34m",  # Blue
    "\033[35m",  # Magenta
    "\033[36m",  # Cyan
    "\033[91m",  # Bright Red
    "\033[92m",  # Bright Green
    "\033[94m",  # Bright Blue
    "\033[95m",  # Bright Magenta
]


def format_text_concise(text: str) -> str:
    """Convert verbose argument descriptions to concise format and show the most relevant operation call."""
    lines = text.split("\n")
    formatted_lines = []

    # Find the most relevant operation call - look for lines containing ttnn operations or math operators
    # Start from the end of the stack trace to find the most user-relevant call
    relevant_file_line = None
    operation_call_line = None

    for i in range(len(lines) - 1, -1, -1):  # Search backwards through stack trace
        line = lines[i]
        line_stripped = line.strip()

        # Look for operation calls (assignments with ttnn calls or math operators)
        if (
            "=" in line_stripped
            and not line_stripped.startswith("File ")
            and ("ttnn." in line_stripped or any(op in line_stripped for op in [" + ", " - ", " * ", " / "]))
        ):
            operation_call_line = line
            # Find the corresponding file line (should be right before this)
            if i > 0 and lines[i - 1].strip().startswith("File "):
                relevant_file_line = lines[i - 1]
            break

    # If no operation found, fallback to finding any meaningful user code (exclude internal/library code)
    if not operation_call_line:
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            line_stripped = line.strip()

            if line_stripped.startswith("File "):
                file_path = line_stripped
                # Skip internal library files, decorators, and framework code
                if not any(
                    internal in file_path
                    for internal in [
                        "/site-packages/",
                        "/lib/python",
                        "decorators.py",
                        "/pytest/",
                        "/pluggy/",
                        "__call__",
                    ]
                ):
                    relevant_file_line = line
                    # Look for the next line that might be a function call
                    if i < len(lines) - 1:
                        next_line = lines[i + 1]
                        if not next_line.strip().startswith("File ") and next_line.strip():
                            operation_call_line = next_line
                    break

    # Add the most relevant file line and operation call
    if relevant_file_line:
        formatted_lines.append(relevant_file_line)
    if operation_call_line:
        formatted_lines.append(operation_call_line)

    # Process argument description lines
    for line in lines:
        line_stripped = line.strip()

        # Convert argument description lines (look for pattern: word : shape = ...)
        if " : " in line_stripped and "shape" in line_stripped:
            key_info = extract_key_info(line_stripped)
            # Preserve original indentation
            original_indent = len(line) - len(line.lstrip())
            formatted_lines.append(" " * original_indent + key_info)

    return "\n".join(formatted_lines)


def resolve_cpp_callstack(callstack: str) -> str:
    """Try to resolve C++ callstack addresses to source locations using addr2line.

    Expects callstack format like: [binary(+0x123)] <- [binary(+0x456)]
    Returns enhanced callstack with file:line if debug symbols are available.
    """
    # Check if this looks like a C++ callstack (contains addresses)
    if not re.search(r"\[.*\(?\+?0x[0-9a-fA-F]+\)?.*\]", callstack):
        return callstack

    # Try to find llvm-addr2line first (better DWARF 5 support), then fall back to addr2line
    addr2line_cmd = None
    for cmd in ["llvm-addr2line-17", "llvm-addr2line", "addr2line"]:
        if shutil.which(cmd):
            addr2line_cmd = cmd
            break

    if not addr2line_cmd:
        return callstack

    # Extract all [binary(+0xoffset)] patterns
    pattern = r"\[([^\(]+)\(\+?(0x[0-9a-fA-F]+)\)\]"
    matches = re.findall(pattern, callstack)

    if not matches:
        return callstack

    resolved_parts = []
    for binary, offset in matches:
        binary = binary.strip()

        # Try to resolve using addr2line
        try:
            result = subprocess.run(
                [addr2line_cmd, "-e", binary, "-f", "-C", offset], capture_output=True, text=True, timeout=1
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")

                # Without -i flag, we get a single pair of (function, location)
                if len(lines) >= 2:
                    function_name = lines[0]
                    location = lines[1]

                    # Check if we got valid debug info (not "??")
                    if location != "??:?" and location != "??:0":
                        # Skip standard library and internal framework code
                        if any(skip in location for skip in ["/include/c++/", "/bits/", "/spdlog/", "/reflect"]):
                            # Fallback to original format
                            resolved_parts.append(f"[{binary}(+{offset})]")
                            continue

                        # Format the location nicely
                        if "/" in location:
                            # Get relative path from tests/ or common location
                            for prefix in ["/tests/", "/ttnn/", "/tt_metal/"]:
                                if prefix in location:
                                    location = location[location.find(prefix) + 1 :]
                                    break
                            else:
                                # Just use the filename
                                location = location.split("/")[-1]

                        resolved_parts.append(f"{location}")
                        continue
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        # Fallback to original format if resolution failed
        resolved_parts.append(f"[{binary}(+{offset})]")

    return " <- ".join(resolved_parts) if resolved_parts else callstack


def extract_key_info(arg_description: str) -> str:
    """Extract key information from verbose argument descriptions."""
    # Example input: "input_tensor : shape = Shape([1, 3, 224, 224]) data_type = DataType::BFLOAT16 memory_config = MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)"
    # Expected output: "input_tensor [1, 3, 224, 224] BFLOAT16 INTERLEAVED DRAM"

    # Extract argument name (before the colon)
    if ":" not in arg_description:
        return arg_description

    arg_name = arg_description.split(":")[0].strip()
    rest = arg_description.split(":", 1)[1]

    info_parts = [arg_name]

    # Extract shape - look for Shape([...])
    shape_match = re.search(r"Shape\(\[([^\]]+)\]\)", rest)
    if shape_match:
        shape = shape_match.group(1)
        info_parts.append(f"[{shape}]")

    # Extract data type - look for DataType::TYPE
    dtype_match = re.search(r"DataType::(\w+)", rest)
    if dtype_match:
        dtype = dtype_match.group(1)
        info_parts.append(dtype)

    # Extract memory layout - look for TensorMemoryLayout::LAYOUT
    layout_match = re.search(r"TensorMemoryLayout::(\w+)", rest)
    if layout_match:
        layout = layout_match.group(1)
        info_parts.append(layout)

    # Extract buffer type - look for BufferType::TYPE
    buffer_match = re.search(r"BufferType::(\w+)", rest)
    if buffer_match:
        buffer_type = buffer_match.group(1)
        info_parts.append(buffer_type)

    return " ".join(info_parts)


def extract_arguments_from_call(call_line: str) -> list[str]:
    """Extract argument names from a function call line."""
    # Match function calls like ttnn.matmul(input_a, input_b) or input_tensor_a + scalar

    # For operation calls like ttnn.matmul(arg1, arg2, ...)
    paren_match = re.search(r"\(([^)]+)\)", call_line)
    if paren_match:
        args_str = paren_match.group(1)
        # Split by comma but handle keyword arguments
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            # Handle keyword arguments like bias=bias_tensor
            if "=" in arg:
                arg = arg.split("=")[-1].strip()
            args.append(arg)
        return args

    # For binary operations like input_tensor_a + scalar
    binary_ops = ["+", "-", "*", "/", "//", "%", "==", "!=", "<", ">", "<=", ">=", "&", "|", "^"]
    for op in binary_ops:
        if f" {op} " in call_line:
            parts = call_line.split(f" {op} ")
            if len(parts) == 2:
                left = parts[0].split("=")[-1].strip() if "=" in parts[0] else parts[0].strip()
                right = parts[1].strip()
                return [left, right]

    return []


def colorize_arguments(text: str) -> str:
    """Add colors to function calls and their corresponding argument descriptions."""
    lines = text.split("\n")

    # Find the operation call (last non-empty line of callstack section)
    call_line = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for lines that contain function calls or operations
        if (
            (
                ("(" in stripped and ")" in stripped)
                or any(op in stripped for op in [" + ", " - ", " * ", " / ", " == ", " != "])
            )
            and not stripped.startswith("File ")
            and not stripped.startswith("return ")
            and not stripped.startswith("exec(")
        ):
            call_line = stripped
            break

    if not call_line:
        return text

    # Extract arguments from the call
    arguments = extract_arguments_from_call(call_line)
    if not arguments:
        return text

    # Create color mapping for arguments and parameter names
    arg_colors = {}  # Maps call arguments (result3, 1.0) to colors
    param_colors = {}  # Maps parameter names (arg_0, arg_1) to colors

    for i, arg in enumerate(arguments):
        color = COLORS[i % len(COLORS)]
        arg_colors[arg] = color
        # Also map positional parameter names
        param_colors[f"arg_{i}"] = color

    # Also look for keyword arguments in the argument descriptions
    # and add any named parameters we find
    for line in lines:
        line_stripped = line.lstrip(" ·")
        # Look for parameter names that aren't arg_0, arg_1 style
        param_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\[=]", line_stripped)
        if param_match:
            param_name = param_match.group(1)
            if param_name not in param_colors and not param_name.startswith("arg_"):
                # Find the position in arguments to assign consistent color
                # This handles keyword arguments like 'tensor', 'device', etc.
                if len(param_colors) < len(COLORS):
                    param_colors[param_name] = COLORS[len(param_colors) % len(COLORS)]

    # Apply colors to the text
    colored_lines = []
    for line in lines:
        colored_line = line

        # Color the function call line
        if call_line in line:
            for arg, color in arg_colors.items():
                # Use word boundaries to match exact argument names
                pattern = r"\b" + re.escape(arg) + r"\b"
                colored_line = re.sub(pattern, f"{color}{arg}{RST}", colored_line)

        # Color the argument descriptions (lines starting with parameter name)
        else:
            line_stripped = line.lstrip(" ·")
            for param_name, color in param_colors.items():
                # Match both verbose format (arg :) and concise format (arg [shape])
                if (
                    line_stripped.startswith(f"{param_name} :")
                    or line_stripped.startswith(f"{param_name} [")
                    or line_stripped.startswith(f"{param_name} =")
                ):
                    # Color the entire line in the argument's color
                    indent_match = re.match(r"^(\s*·?\s*)", line)
                    indent = indent_match.group(1) if indent_match else ""
                    content = line_stripped
                    # Color the parameter name at the start
                    content = re.sub(r"^(" + re.escape(param_name) + r")", f"{color}\\1{RST}", content)
                    colored_line = indent + content
                    break

        colored_lines.append(colored_line)

    return "\n".join(colored_lines)


def preserve_indentation_serializer(value) -> str:
    """Custom serializer that preserves indentation and adds colors to multiline strings."""
    if isinstance(value, str) and "\n" in value:
        # First apply colorization
        colored_value = colorize_arguments(value)

        # Then replace leading spaces with a single dot followed by spaces
        lines = colored_value.split("\n")
        preserved_lines = []
        for line in lines:
            # Count leading spaces and replace with one dot + remaining spaces
            leading_spaces = len(line) - len(line.lstrip(" "))
            if leading_spaces > 0:
                # Use one dot followed by spaces to show indentation
                preserved_line = "·" + (" " * (leading_spaces - 1)) + line.lstrip(" ")
                preserved_lines.append(preserved_line)
            else:
                preserved_lines.append(line)
        return "\n".join(preserved_lines)
    return str(value)


script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data"],
)


@dataclass
class DumpOpsData:
    dev_core: str = triage_field("Dev/Core")
    operation: str = triage_field("Operation")
    callstack_and_args: str = triage_field("Callstack / Arguments", preserve_indentation_serializer)


def format_ops_table(ops_data: list[DumpOpsData], use_mapping: bool = False) -> str:
    """Format operations data as a table."""
    if not ops_data:
        return "No operations found."

    if use_mapping:
        # Format with callstack and args
        result = []
        for op in ops_data:
            location_str = op.location.to_str("logical")
            result.append(f"Core Location: {location_str}")
            result.append(f"RISC Type: {op.risc_type}")
            result.append(f"Kernel Config Host ID: {op.host_assigned_id}")
            if op.host_info and op.host_info != str(op.host_assigned_id):
                result.append("****HOST INFO****")
                result.append(op.host_info.strip())
                result.append("**** END HOST INFO ****")
            result.append("")  # Empty line between entries
        return "\n".join(result)
    else:
        # Original table format
        header = f"{'Core Location':<15} {'RISC Type':<10} {'Host ID':<10}"
        separator = "-" * 37

        rows = []
        for op in ops_data:
            location_str = op.location.to_str("logical")
            rows.append(f"{location_str:<15} {op.risc_type:<10} {op.host_assigned_id:<10}")

        return "\n".join([header, separator] + rows)


def fetch_operations_from_serialized_files(inspector_path="generated/inspector") -> dict:
    """Read operations from serialized capnp files when RPC is not available."""
    import os
    import capnp

    # Load the capnp schema
    capnp_file = os.path.join(os.path.dirname(__file__), "../../tt_metal/impl/debug/inspector/rpc.capnp")
    if not os.path.exists(capnp_file):
        print(f"[Warning] Cannot find capnp schema at {capnp_file}")
        return {}

    rpc_capnp = capnp.load(capnp_file)

    # Look for serialized operations file
    operations_file = os.path.join(inspector_path, "getOperations.capnp.bin")

    if os.path.exists(operations_file):
        try:
            with open(operations_file, "rb") as f:
                # Read packed message
                operations_response = rpc_capnp.Inspector.GetOperationsResults.read_packed(f)

                # Convert to mapping dict keyed by device_operation_id
                mapping = {}
                for op in operations_response.operations:
                    # Skip host-only operations
                    if op.deviceOperationId != "none":
                        mapping[op.deviceOperationId] = {
                            "device_operation_id": op.deviceOperationId,
                            "operation_name": op.operationName,
                            "call_stack": op.callstack,
                            "arguments": op.arguments,
                        }

                print(f"[Info] Loaded {len(mapping)} operations from serialized file")
                return mapping

        except Exception as e:
            print(f"[Warning] Failed to read serialized operations file: {e}")
            return {}
    else:
        print(f"[Info] No serialized operations file found at {operations_file}")
        return {}


def fetch_operations_from_inspector(inspector) -> dict:
    """Fetch operations from Inspector RPC and convert to mapping format."""
    try:
        # Try to get operations
        try:
            operations_response = inspector.getOperations()
            operations = operations_response.operations

            # Convert to mapping dict keyed by device_operation_id
            mapping = {}
            for op in operations:
                # Skip host-only operations
                if op.deviceOperationId != "none":
                    mapping[op.deviceOperationId] = {
                        "device_operation_id": op.deviceOperationId,
                        "operation_name": op.operationName,
                        "callstack": op.callstack,
                        "arguments": op.arguments,
                    }
            return mapping
        except AttributeError as e:
            # Inspector doesn't have getOperations method (old version or not RPC)
            print(f"[Warning] Inspector doesn't support getOperations: {e}")
            return {}

    except Exception as e:
        # If we can't get operations from Inspector, return empty dict
        print(f"[Warning] Could not fetch operations from Inspector: {e}")
        return {}


def dump_ops(
    device: Device,
    dispatcher_data: DispatcherData,
    max_width: int = 100,
    verbose: bool = False,
    summary: bool = False,
    inspector_data=None,
) -> tuple[list[DumpOpsData], list[tuple[int, str]]]:
    """Extract core location and host ID for all operations.

    Returns:
        tuple: (list of DumpOpsData, list of (host_id, operation_name) tuples)
    """
    blocks_to_test = ["functional_workers", "eth"]
    host_id_op_names: list[tuple[int, str]] = []

    # Get operations from Inspector RPC
    host_id_mapping = fetch_operations_from_inspector(inspector_data) if inspector_data else {}

    seen_host_ids = set()

    # Aggregate cores by operation ID
    # Maps operation_id -> list of (device_id, location) tuples
    op_to_cores: dict[int, list[tuple[int, str]]] = {}

    for block_to_test in blocks_to_test:
        for location in device.get_block_locations(block_to_test):
            noc_block = device.get_block(location)

            # We support only idle eth blocks for now
            if noc_block.block_type == "eth" and noc_block not in device.idle_eth_blocks:
                continue

            # Check all RISC cores but only add one entry per location since data repeats
            kernel_config_host_id = None
            for risc_name in noc_block.risc_names:
                dispatcher_core_data = dispatcher_data.get_core_data(location, risc_name)

                # Only include cores that have valid kernel config host assigned IDs (not -1 or 0)
                if dispatcher_core_data.host_assigned_id not in [-1, 0]:
                    kernel_config_host_id = dispatcher_core_data.host_assigned_id
                    break  # Data is the same across all RISCs, so use first valid one

            # If we found a valid kernel_config_host_id, track this core
            if kernel_config_host_id is not None:
                # Track unique host IDs and their operation names
                if kernel_config_host_id not in seen_host_ids and kernel_config_host_id > 0:
                    seen_host_ids.add(kernel_config_host_id)

                    # device_operation_id matches host_assigned_id directly
                    operation_id_key = str(kernel_config_host_id)

                    if operation_id_key in host_id_mapping:
                        mapping = host_id_mapping[operation_id_key]
                        if mapping.get("device_operation_id") != "none":
                            op_name = mapping.get("operation_name", "unknown_op")
                            host_id_op_names.append((kernel_config_host_id, op_name))

                # Add this core to the list for this operation
                if kernel_config_host_id not in op_to_cores:
                    op_to_cores[kernel_config_host_id] = []
                op_to_cores[kernel_config_host_id].append((device._id, location.to_str("logical")))

    # Now create aggregated results - one entry per unique operation
    result: list[DumpOpsData] = []

    for kernel_config_host_id, cores in sorted(op_to_cores.items()):
        operation_id_key = str(kernel_config_host_id)

        # Get operation info from mapping if available
        operation_name = None
        callstack = ""
        args = ""
        if operation_id_key in host_id_mapping:
            mapping = host_id_mapping[operation_id_key]
            operation_name = mapping.get("operation_name", None)
            callstack = mapping.get("callstack", "")
            args = mapping.get("arguments", "")

        # Format operation name for the Operation column
        if operation_name:
            operation_display = operation_name
        else:
            # No mapping available, just show the ID
            operation_display = f"ID: {kernel_config_host_id}"

        # Format callstack and arguments for the combined column
        callstack_args_lines = []

        # Add callstack if available
        if callstack:
            # Try to resolve C++ addresses to source locations if debug symbols are available
            resolved_callstack = resolve_cpp_callstack(callstack)

            # Parse and format callstack frames - each frame on its own line
            # Expected format: "#0 file.py:42 #1 file.py:81" or "#0 func [binary(+0x123)] #1 ..."
            # Also handle old format: "func1 <- func2 <- func3"
            import re

            frame_pattern = r"#\d+\s+[^#]+"
            frames = re.findall(frame_pattern, resolved_callstack)

            if frames:
                # New numbered format - filter out decorator frames and runtime startup code
                # Keep only user code and meaningful library frames
                def is_unhelpful_frame(frame):
                    """Check if frame is internal decorator or runtime startup code"""
                    # Filter out decorator template frames
                    if "decorators.hpp" in frame:
                        return True
                    # Filter out C runtime startup code
                    if any(pattern in frame for pattern in ["_start", "__libc_start_main", "libc.so.6"]):
                        return True
                    # Filter out unresolved frames from the binary itself (startup code)
                    # These look like: [build/test/.../binary(+0x12345)]
                    if "run_operation_chain_cpp(+" in frame and ".cpp:" not in frame:
                        return True
                    return False

                filtered_frames = [f for f in frames if not is_unhelpful_frame(f)]
                if filtered_frames:
                    callstack_args_lines.append("Callstack:")
                    for frame in filtered_frames:
                        # Add with two-space indent and wrap if too long
                        frame_stripped = frame.strip()
                        if len(frame_stripped) + 2 <= max_width:
                            callstack_args_lines.append(f"  {frame_stripped}")
                        else:
                            # Wrap the frame
                            wrapped = textwrap.fill(
                                frame_stripped,
                                width=max_width,
                                initial_indent="  ",
                                subsequent_indent="    ",
                                break_long_words=False,
                                break_on_hyphens=False,
                            )
                            callstack_args_lines.append(wrapped)
            elif "<-" in resolved_callstack:
                # Old format with <- separator - split and number frames, filter unhelpful frames
                def is_unhelpful_frame_old(frame):
                    """Check if frame is internal decorator or runtime startup code"""
                    # Filter out decorator template frames
                    if "decorators.hpp" in frame:
                        return True
                    # Filter out C runtime startup code
                    if any(pattern in frame for pattern in ["_start", "__libc_start_main", "libc.so.6"]):
                        return True
                    # Filter out unresolved binary frames
                    if "run_operation_chain_cpp(+" in frame and ".cpp:" not in frame:
                        return True
                    return False

                old_frames = [f.strip() for f in resolved_callstack.split("<-") if not is_unhelpful_frame_old(f)]
                if old_frames:
                    callstack_args_lines.append("Callstack:")
                    for i, frame in enumerate(old_frames):
                        frame_text = f"#{i} {frame}"
                        if len(frame_text) + 2 <= max_width:
                            callstack_args_lines.append(f"  {frame_text}")
                        else:
                            # Wrap the frame
                            wrapped = textwrap.fill(
                                frame_text,
                                width=max_width,
                                initial_indent="  ",
                                subsequent_indent="    ",
                                break_long_words=False,
                                break_on_hyphens=False,
                            )
                            callstack_args_lines.append(wrapped)
            else:
                # Single frame or unparseable format - wrap if needed
                if len(resolved_callstack) + 11 <= max_width:  # 11 = len("Callstack: ")
                    callstack_args_lines.append(f"Callstack: {resolved_callstack}")
                else:
                    wrapped = textwrap.fill(
                        resolved_callstack,
                        width=max_width,
                        initial_indent="Callstack: ",
                        subsequent_indent="           ",
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    callstack_args_lines.append(wrapped)

        # Add arguments based on mode
        if args:
            if summary:
                # Summary mode: extract and show each argument on its own line
                import re

                # Split arguments by top-level commas (but not commas inside parentheses/brackets)
                # Simple approach: find commas that are not inside any brackets
                bracket_depth = 0
                paren_depth = 0
                arg_parts = []
                current_arg = []

                for char in args:
                    if char == "(" or char == "[":
                        paren_depth += 1
                        bracket_depth += 1
                        current_arg.append(char)
                    elif char == ")" or char == "]":
                        paren_depth -= 1
                        bracket_depth -= 1
                        current_arg.append(char)
                    elif char == "," and bracket_depth == 0:
                        # This is a top-level comma - split here
                        arg_parts.append("".join(current_arg).strip())
                        current_arg = []
                    else:
                        current_arg.append(char)

                # Don't forget the last argument
                if current_arg:
                    arg_parts.append("".join(current_arg).strip())

                # Parse each argument
                arguments = []
                for arg in arg_parts:
                    if not arg:
                        continue

                    # Check if this is a tensor (contains "logical_shape=Shape")
                    shape_match = re.search(r"logical_shape=Shape\(\[([^\]]+)\]\)", arg)
                    if shape_match:
                        shape = shape_match.group(1)
                        properties = []

                        # Extract data type
                        dtype_match = re.search(r"dtype=DataType::(\w+)", arg)
                        if dtype_match:
                            properties.append(dtype_match.group(1))

                        # Extract memory layout
                        layout_match = re.search(r"memory_layout=TensorMemoryLayout::(\w+)", arg)
                        if layout_match:
                            properties.append(layout_match.group(1))

                        # Extract buffer type
                        buffer_match = re.search(r"buffer_type=BufferType::(\w+)", arg)
                        if buffer_match:
                            properties.append(buffer_match.group(1))

                        if properties:
                            arguments.append(f"  Tensor[{shape}] ({', '.join(properties)})")
                        else:
                            arguments.append(f"  Tensor[{shape}]")
                    else:
                        # Non-tensor argument - show compact representation
                        # Could be a scalar, enum, etc.
                        # Use max_width to determine truncation limit
                        arg_with_indent = f"  {arg}"
                        if len(arg_with_indent) <= max_width:
                            arguments.append(arg_with_indent)
                        else:
                            # Truncate to fit within max_width, accounting for "..." and indent
                            truncate_len = max_width - 5  # 2 for indent + 3 for "..."
                            if truncate_len > 0:
                                arguments.append(f"  {arg[:truncate_len]}...")
                            else:
                                arguments.append(f"  {arg[:max_width]}")

                if arguments:
                    callstack_args_lines.append("Arguments:")
                    for arg_line in arguments:
                        # Wrap argument lines if they exceed max_width
                        if len(arg_line) <= max_width:
                            callstack_args_lines.append(arg_line)
                        else:
                            wrapped = textwrap.fill(
                                arg_line.strip(),
                                width=max_width,
                                initial_indent="  ",
                                subsequent_indent="    ",
                                break_long_words=False,
                                break_on_hyphens=False,
                            )
                            callstack_args_lines.append(wrapped)
                else:
                    # Use max_width to determine fallback truncation limit
                    # Account for "Arguments: " (11 chars) and "..." (3 chars)
                    truncate_len = max_width - 14
                    if truncate_len > 0 and len(args) > truncate_len:
                        fallback_text = f"Arguments: {args[:truncate_len]}..."
                    else:
                        fallback_text = f"Arguments: {args}"

                    if len(fallback_text) <= max_width:
                        callstack_args_lines.append(fallback_text)
                    else:
                        # Note: "Arguments: " is 11 chars, but continuation should not get bullet
                        # So we use more indentation for continuation
                        wrapped = textwrap.fill(
                            args,
                            width=max_width,
                            initial_indent="Arguments: ",
                            subsequent_indent="           ",  # 11 spaces - will be preserved as-is (>4)
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                        callstack_args_lines.append(wrapped)

        callstack_args_info = "\n".join(callstack_args_lines) if callstack_args_lines else "No details"

        # Format the list of cores for this operation
        if len(cores) == 1:
            dev_core = f"{cores[0][0]} / {cores[0][1]}"
        else:
            # Multiple cores - show count and list with max 4 core IDs per line
            core_strs = [f"{dev_id}/{loc}" for dev_id, loc in cores]
            core_lines = []
            for i in range(0, len(core_strs), 4):
                core_lines.append(", ".join(core_strs[i : i + 4]))

            # In summary mode, limit to 12 lines max
            if summary and len(core_lines) > 12:
                core_lines = core_lines[:12]
                core_lines.append("...")

            if len(core_lines) == 1:
                dev_core = f"{len(cores)} cores: {core_lines[0]}"
            else:
                # Multi-line formatting with proper indentation
                dev_core = f"{len(cores)} cores:\n  " + "\n  ".join(core_lines)

        ops_data = DumpOpsData(
            dev_core=dev_core,
            operation=operation_display,
            callstack_and_args=callstack_args_info,
        )
        result.append(ops_data)

    return result, host_id_op_names


def run(args, context: Context):
    """Run the dump_ops script."""
    max_width = int(args["--max-width"]) if args["--max-width"] else 100
    # Check for generate-test flag - args returns None if not present
    generate_test = bool(args["--generate-test"])
    # Always use summary mode
    summary = True
    verbose_mode = False
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    inspector_data = get_inspector_data(args, context)

    # Handle device iteration directly to avoid automatic "Dev" column
    all_ops_data = []
    all_host_id_op_names = []
    for device in run_checks.devices:
        device_ops, host_id_op_names = dump_ops(
            device, dispatcher_data, max_width, verbose_mode, summary, inspector_data
        )
        all_ops_data.extend(device_ops)
        all_host_id_op_names.extend(host_id_op_names)

    # Save host_id_op_names to a module-level variable for access in __main__
    import dump_ops as this_module

    this_module._collected_host_id_op_names = all_host_id_op_names

    # Generate tests if requested
    if generate_test:
        generate_tests_for_operations(all_host_id_op_names, inspector_data)

    return all_ops_data


if __name__ == "__main__":
    run_script()
