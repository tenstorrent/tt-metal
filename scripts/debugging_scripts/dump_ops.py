#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_ops [--max-width=<width>] [--verbose]

Options:
    --max-width=<width>    Maximum column width for wrapping text [default: 120]
    --verbose              Show detailed argument information (default: concise)

Description:
    Dumps core location and kernel config host-assigned ID for all operations in a table format.
    If Inspector RPC is available, shows operation names and details.
    By default shows concise argument info, use --verbose for full details.
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

import re, textwrap

script_config = ScriptConfig(
    data_provider=False,
    depends=["inspector_data", "dispatcher_data"],
)

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
    host_info: str = triage_field("Operation", preserve_indentation_serializer)


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
            result.append(f"Kernel Config Host ID: {op.kernel_config_host_assigned_id}")
            if op.host_info and op.host_info != str(op.kernel_config_host_assigned_id):
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
            rows.append(f"{location_str:<15} {op.risc_type:<10} {op.kernel_config_host_assigned_id:<10}")

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
                        "call_stack": op.callstack,
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
    inspector_data=None,
) -> tuple[list[DumpOpsData], list[tuple[int, str]]]:
    """Extract core location and host ID for all operations.

    Returns:
        tuple: (list of DumpOpsData, list of (host_id, operation_name) tuples)
    """
    blocks_to_test = ["functional_workers", "eth"]
    result: list[DumpOpsData] = []
    host_id_op_names: list[tuple[int, str]] = []

    # Get operations from Inspector RPC
    host_id_mapping = fetch_operations_from_inspector(inspector_data) if inspector_data else {}

    seen_host_ids = set()

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
                if dispatcher_core_data.kernel_config_host_assigned_id not in [-1, 0]:
                    kernel_config_host_id = dispatcher_core_data.kernel_config_host_assigned_id
                    break  # Data is the same across all RISCs, so use first valid one

            # If we found a valid kernel_config_host_id, add one entry for this location
            if kernel_config_host_id is not None:
                # Track unique host IDs and their operation names
                if kernel_config_host_id not in seen_host_ids and kernel_config_host_id > 0:
                    seen_host_ids.add(kernel_config_host_id)

                    # device_operation_id matches kernel_config_host_assigned_id directly
                    operation_id_key = str(kernel_config_host_id)

                    if operation_id_key in host_id_mapping:
                        mapping = host_id_mapping[operation_id_key]
                        if mapping.get("device_operation_id") != "none":
                            op_name = mapping.get("operation_name", "unknown_op")
                            host_id_op_names.append((kernel_config_host_id, op_name))

                # device_operation_id matches kernel_config_host_assigned_id directly
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

                # Format host info based on whether we have mapping data
                if operation_name:
                    # Show operation name as primary info
                    if verbose:
                        # In verbose mode, show operation name + args + callstack
                        host_info_lines = [operation_name]

                        # Add arguments if available (they contain tensor shapes)
                        if args:
                            # Format arguments nicely
                            arg_lines = args.split("\n")
                            for arg_line in arg_lines:
                                if arg_line.strip():
                                    # Indent arguments
                                    host_info_lines.append(f"  {arg_line}")

                        # Add callstack if available
                        if callstack:
                            host_info_lines.append("  Callstack:")
                            # Limit callstack to first 3 frames in table view
                            stack_lines = callstack.split("\n")[:3]
                            for stack_line in stack_lines:
                                if stack_line.strip():
                                    host_info_lines.append(f"    {stack_line.strip()}")
                            if len(callstack.split("\n")) > 3:
                                host_info_lines.append("    ...")

                        # Wrap long lines
                        wrapped_lines = []
                        for line in host_info_lines:
                            if len(line) <= max_width:
                                wrapped_lines.append(line)
                            else:
                                # Detect original indentation and preserve it
                                original_indent = len(line) - len(line.lstrip())
                                indent_str = line[:original_indent]

                                # Wrap this line, preserving original indentation for continuation
                                wrapped = textwrap.fill(line, width=max_width, subsequent_indent=indent_str + "  ")
                                wrapped_lines.append(wrapped)

                        host_info = "\n".join(wrapped_lines)
                    else:
                        # In concise mode, show operation name and first tensor shape if available
                        if args and "Tensor[shape=" in args:
                            # Extract first tensor shape
                            import re

                            shape_match = re.search(r"shape=\(([^)]+)\)", args)
                            if shape_match:
                                host_info = f"{operation_name} [{shape_match.group(1)}]"
                            else:
                                host_info = operation_name
                        else:
                            host_info = operation_name
                else:
                    # No mapping available, just show the ID
                    host_info = str(kernel_config_host_id)

                # Combine device ID and core location into a single string
                dev_core = f"{device._id} / {location.to_str('logical')}"

                ops_data = DumpOpsData(
                    dev_core=dev_core,
                    host_info=host_info,
                )
                result.append(ops_data)

    return result, host_id_op_names


def run(args, context: Context):
    """Run the dump_ops script."""
    max_width = int(args["--max-width"]) if args["--max-width"] else 100
    verbose = args["--verbose"]
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    inspector_data = get_inspector_data(args, context)

    # Handle device iteration directly to avoid automatic "Dev" column
    all_ops_data = []
    all_host_id_op_names = []
    for device in run_checks.devices:
        device_ops, host_id_op_names = dump_ops(device, dispatcher_data, max_width, verbose, inspector_data)
        all_ops_data.extend(device_ops)
        all_host_id_op_names.extend(host_id_op_names)

    # Save host_id_op_names to a module-level variable for access in __main__
    import dump_ops as this_module

    this_module._collected_host_id_op_names = all_host_id_op_names

    return all_ops_data


if __name__ == "__main__":
    run_script()
