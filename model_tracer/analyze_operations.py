# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
TTNN Operations Master File Analyzer

This script analyzes the ttnn_operations_master.json file and allows you to:
- View all configurations for a specific operation
- Search for operations by partial name
- List all available operations
- Show operation statistics

Usage:
    python analyze_operations.py add                    # Show all ttnn::add configurations
    python analyze_operations.py linear                 # Show all ttnn::linear configurations
    python analyze_operations.py --list                 # List all operations
    python analyze_operations.py --stats                # Show statistics
    python analyze_operations.py experimental.add       # Show ttnn::experimental::add
"""

import json
import argparse
import os
from typing import Dict, List, Any
import sys
from datetime import datetime


def load_master_file(file_path: str) -> Dict[str, Any]:
    """Load and validate the master JSON file"""
    if not os.path.exists(file_path):
        print(f"❌ Master file not found: {file_path}")
        print("💡 Run generic_ops_tracer.py first to generate the master file")
        sys.exit(1)

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if "operations" not in data:
            print(f"❌ Invalid master file format - missing 'operations' key")
            sys.exit(1)

        return data

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)


def find_matching_operations(operations: Dict[str, Any], search_term: str) -> List[str]:
    """Find operations that match the search term"""
    matches = []

    # Normalize search term - add ttnn:: prefix if not present
    if not search_term.startswith("ttnn::"):
        search_patterns = [
            f"ttnn::{search_term}",
            f"ttnn::experimental::{search_term}",
            f"ttnn::transformer::{search_term}",
            f"ttnn::moreh::{search_term}",
            f"ttnn::kv_cache::{search_term}",
            f"ttnn::prim::{search_term}",
        ]
    else:
        search_patterns = [search_term]

    # Find exact matches first - these take absolute priority
    exact_matches = []
    for pattern in search_patterns:
        if pattern in operations:
            exact_matches.append(pattern)

    # If we have exact matches, return only those
    if exact_matches:
        return sorted(exact_matches)

    # Only if no exact matches exist, look for partial matches
    partial_matches = []
    search_lower = search_term.lower()

    for op_name in operations.keys():
        # More precise partial matching - avoid matching ttnn::add_ when looking for ttnn::add
        op_name_lower = op_name.lower()

        # Check if search term appears as a complete word/component
        if search_lower in op_name_lower:
            # Additional check: if searching for "add", don't match "add_"
            # unless the search term itself has underscores
            if "_" not in search_term and op_name_lower.endswith("_"):
                # Skip operations ending with _ if we're not specifically searching for them
                continue
            partial_matches.append(op_name)

    return sorted(partial_matches)


def print_operation_configs(
    op_name: str, configurations: List[Any], show_details: bool = True, debug_mode: bool = False
):
    """Pretty print operation configurations"""
    print(f"\n🔧 Operation: {op_name}")
    print(f"📊 Configurations: {len(configurations)}")
    print("=" * 80)

    if not show_details:
        print("💡 Use --details flag to see full configuration data")
        return

    for i, config in enumerate(configurations, 1):
        print(f"\n📋 Configuration {i}:")
        print("-" * 40)

        if isinstance(config, list):
            if debug_mode:
                # Show ALL arguments in debug mode
                for j, arg in enumerate(config):
                    formatted_arg = format_argument(arg)
                    print(f"  arg{j}: {formatted_arg}")
            else:
                # Filter out nullopt/None arguments (normal mode)
                relevant_args = []
                for j, arg in enumerate(config):
                    if isinstance(arg, dict):
                        # Check if the argument value is nullopt or None-like
                        arg_key = f"arg{j}"
                        if arg_key in arg:
                            arg_value = arg[arg_key]
                            # Skip nullopt, None, and unsupported types
                            if (
                                arg_value == "nullopt"
                                or arg_value is None
                                or (
                                    isinstance(arg_value, str)
                                    and ("unsupported type" in arg_value or "std::reference_wrapper" in arg_value)
                                )
                            ):
                                continue
                        relevant_args.append((j, arg))

                # Print only relevant arguments
                if relevant_args:
                    for j, arg in relevant_args:
                        formatted_arg = format_argument(arg)
                        print(f"  arg{j}: {formatted_arg}")
                else:
                    print("  (All arguments are default/internal)")
        else:
            print(f"  {format_argument(config)}")


def format_shard_config(memory_config: Dict[str, Any]) -> str:
    """Format shard configuration details if present"""
    # Try nd_shard_spec first (preferred), then shard_spec
    shard_spec = memory_config.get("nd_shard_spec")
    if not shard_spec or shard_spec == "std::nullopt":
        shard_spec = memory_config.get("shard_spec")

    if not shard_spec or shard_spec == "std::nullopt":
        return ""

    parts = []

    # Extract shard_shape
    shard_shape = shard_spec.get("shard_shape") or shard_spec.get("shape")
    if shard_shape:
        # Handle both list format [224, 32] and string format "{224, 32}"
        if isinstance(shard_shape, str):
            import re

            # Extract numbers from string like "{224, 32}"
            shape_match = re.search(r"{\s*(\d+)\s*,\s*(\d+)\s*}", shard_shape)
            if shape_match:
                shard_shape = [int(shape_match.group(1)), int(shape_match.group(2))]
            else:
                # Try bracket format
                shape_match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", shard_shape)
                if shape_match:
                    shard_shape = [int(shape_match.group(1)), int(shape_match.group(2))]

        if isinstance(shard_shape, list):
            parts.append(f"shard_shape={shard_shape}")

    # Extract grid
    grid = shard_spec.get("grid")
    if grid:
        grid_str = format_grid(grid)
        if grid_str:
            parts.append(f"grid={grid_str}")

    # Extract orientation
    orientation = shard_spec.get("orientation", "")
    if orientation:
        orientation = orientation.replace("ShardOrientation::", "")
        parts.append(f"orientation={orientation}")

    if parts:
        return "shard(" + ", ".join(parts) + ")"
    return ""


def format_grid(grid: Any) -> str:
    """Format grid configuration"""
    if isinstance(grid, list):
        # Handle both simple [{"x":0,"y":0}, {"x":7,"y":7}]
        # and complex [[{...}, {...}], [{...}, {...}]] formats
        if len(grid) == 2:
            if isinstance(grid[0], dict) and "x" in grid[0]:
                # Simple format: [start, end]
                return f"[({grid[0]['x']},{grid[0]['y']})→({grid[1]['x']},{grid[1]['y']})]"
            elif isinstance(grid[0], list):
                # Complex format: multiple ranges
                ranges = []
                for range_pair in grid:
                    if len(range_pair) == 2:
                        start, end = range_pair
                        ranges.append(f"({start['x']},{start['y']})→({end['x']},{end['y']})")
                return f"[{', '.join(ranges)}]"
        return str(grid)
    elif isinstance(grid, str):
        # Try to parse string representation
        import re

        # Match patterns like "[{x:0,y:0} - {x:7,y:6}]"
        match = re.search(r"x[:\s]*(\d+).*?y[:\s]*(\d+).*?x[:\s]*(\d+).*?y[:\s]*(\d+)", grid)
        if match:
            x1, y1, x2, y2 = match.groups()
            return f"[({x1},{y1})→({x2},{y2})]"
    return str(grid)


def format_argument(arg: Any, max_depth: int = 2, current_depth: int = 0) -> str:
    """Format argument for readable display"""
    if current_depth > max_depth:
        return "..."

    if isinstance(arg, dict):
        if "Tensor" in arg:
            tensor_info = arg["Tensor"]
            if "tensor_spec" in tensor_info:
                spec = tensor_info["tensor_spec"]
                shape = spec.get("logical_shape", "Unknown shape")

                # Extract dtype
                tensor_layout = spec.get("tensor_layout", {})
                dtype = tensor_layout.get("dtype", "Unknown dtype")

                # Extract memory config info
                memory_config = tensor_layout.get("memory_config", {})
                memory_layout = memory_config.get("memory_layout", "").replace("TensorMemoryLayout::", "")
                buffer_type = memory_config.get("buffer_type", "").replace("BufferType::", "")

                # Format memory info
                memory_info = f"{buffer_type}_{memory_layout}" if buffer_type and memory_layout else "Unknown memory"

                # Extract shard configuration details
                shard_details = format_shard_config(memory_config)

                result = f"Tensor(shape={shape}, dtype={dtype.replace('DataType::', '')}, memory={memory_info}"
                if shard_details:
                    result += f", {shard_details}"
                result += ")"

                return result
            return "Tensor(...)"
        elif "Scalar" in arg:
            return f"Scalar({arg['Scalar']})"
        elif "MemoryConfig" in arg:
            mem_config = arg["MemoryConfig"]
            memory_layout = mem_config.get("memory_layout", "").replace("TensorMemoryLayout::", "")
            buffer_type = mem_config.get("buffer_type", "").replace("BufferType::", "")

            memory_info = f"{buffer_type}_{memory_layout}" if buffer_type and memory_layout else "Unknown memory"

            # Extract shard configuration details
            shard_details = format_shard_config(mem_config)

            result = f"MemoryConfig({memory_info}"
            if shard_details:
                result += f", {shard_details}"
            result += ")"

            return result
        elif "Shape" in arg:
            shape_data = arg["Shape"]
            if isinstance(shape_data, list):
                return f"Shape({shape_data})"
            elif isinstance(shape_data, dict) and "shape" in shape_data:
                return f"Shape({shape_data['shape']})"
            else:
                return f"Shape({shape_data})"
        elif "UnparsedElement" in arg:
            # Handle elements that couldn't be parsed during serialization
            unparsed = arg["UnparsedElement"]
            error = unparsed.get("error", "Unknown error")
            element_info = unparsed.get("element_info", "")

            # Try to recover data from element_info if it's valid JSON
            if element_info.startswith("{") and '"Tensor"' in element_info:
                try:
                    # Attempt to parse the JSON string to recover tensor data
                    import json
                    import re

                    # First try direct parsing
                    try:
                        recovered_data = json.loads(element_info)
                    except json.JSONDecodeError:
                        # Try to fix common C++ representation issues
                        fixed_json_str = element_info
                        # Fix C++ style braces in values like "{32, 32}" -> "[32, 32]"
                        fixed_json_str = re.sub(r':\s*"{\s*([^}]+)\s*}"', r': "[\1]"', fixed_json_str)
                        # Fix grid format: "grid":{[...], [...]} -> "grid":[[...], [...]]
                        fixed_json_str = re.sub(
                            r'"grid"\s*:\s*\{(\[.*?\](?:\s*,\s*\[.*?\])*)\}', r'"grid":[\1]', fixed_json_str
                        )
                        # Fix grid ranges like [{"x":0,"y":0} - {"x":7,"y":7}] -> [{"x":0,"y":0}, {"x":7,"y":7}]
                        fixed_json_str = re.sub(r"(\{[^}]+\})\s*-\s*(\{[^}]+\})", r"\1, \2", fixed_json_str)

                        recovered_data = json.loads(fixed_json_str)

                    # If it contains tensor info, extract and format it
                    if isinstance(recovered_data, dict):
                        for key, value in recovered_data.items():
                            if isinstance(value, dict) and "Tensor" in value:
                                tensor_spec = value["Tensor"].get("tensor_spec", {})
                                shape = tensor_spec.get("logical_shape", "Unknown shape")
                                tensor_layout = tensor_spec.get("tensor_layout", {})
                                dtype = tensor_layout.get("dtype", "Unknown dtype").replace("DataType::", "")

                                # Extract memory config
                                memory_config = tensor_layout.get("memory_config", {})
                                memory_layout = memory_config.get("memory_layout", "").replace(
                                    "TensorMemoryLayout::", ""
                                )
                                buffer_type = memory_config.get("buffer_type", "").replace("BufferType::", "")

                                memory_info = (
                                    f"{buffer_type}_{memory_layout}"
                                    if buffer_type and memory_layout
                                    else "Unknown memory"
                                )

                                # Extract shard configuration details
                                shard_details = format_shard_config(memory_config)

                                result = f"Tensor(shape={shape}, dtype={dtype}, memory={memory_info}"
                                if shard_details:
                                    result += f", {shard_details}"
                                result += ")"

                                return result

                except Exception:
                    pass

            # Try to extract useful info from the element_info string (fallback)
            if "logical_shape" in element_info:
                import re

                # Extract shape from the JSON-like string
                shape_match = re.search(r'"logical_shape":\[([^\]]+)\]', element_info)
                if shape_match:
                    shape_str = shape_match.group(1)
                    try:
                        shape = [int(x.strip()) for x in shape_str.split(",")]
                        return f"Tensor(shape={shape}, PARSE_ERROR)"
                    except:
                        pass

            return f"UnparsedElement(error: {error[:50]}...)"
        elif len(arg) <= 3:
            # Small dict, show contents
            items = []
            for k, v in arg.items():
                formatted_v = format_argument(v, max_depth, current_depth + 1)
                items.append(f"{k}: {formatted_v}")
            return "{" + ", ".join(items) + "}"
        else:
            return f"Dict({len(arg)} keys)"
    elif isinstance(arg, list):
        if len(arg) <= 3:
            items = [format_argument(item, max_depth, current_depth + 1) for item in arg]
            return f"[{', '.join(items)}]"
        else:
            return f"List({len(arg)} items)"
    elif isinstance(arg, str):
        return f'"{arg}"'
    else:
        return str(arg)


def show_statistics(data: Dict[str, Any]):
    """Show comprehensive statistics about the master file"""
    operations = data.get("operations", {})
    metadata = data.get("metadata", {})

    print("\n📊 MASTER FILE STATISTICS")
    print("=" * 80)

    # Basic metadata
    print(f"📁 File Metadata:")
    print(f"   🔧 Total Operations: {len(operations)}")
    print(
        f"   ⚙️ Total Configurations: {sum(len(op_data.get('configurations', [])) for op_data in operations.values())}"
    )
    print(f"   📊 Operations Processed: {metadata.get('total_operations', 0)}")
    print(f"   🎯 Models Processed: {len(metadata.get('models', []))}")
    print(f"   🕐 Last Updated: {metadata.get('last_updated', 'Unknown')}")

    # Models list
    models = metadata.get("models", [])
    if models:
        print(f"\n📋 Processed Models:")
        for model in models:
            print(f"   • {model}")

    # Operation statistics
    op_stats = []
    for op_name, op_data in operations.items():
        config_count = len(op_data.get("configurations", []))
        op_stats.append((op_name, config_count))

    op_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔧 Top Operations by Configuration Count:")
    for op_name, count in op_stats[:15]:
        print(f"   {op_name}: {count} configs")

    # Operation type breakdown
    categories = {}
    for op_name, _ in op_stats:
        if "::experimental::" in op_name:
            category = "experimental"
        elif "::transformer::" in op_name:
            category = "transformer"
        elif "::moreh::" in op_name:
            category = "moreh"
        elif op_name.startswith("ttnn::"):
            category = "core"
        else:
            category = "other"

        categories[category] = categories.get(category, 0) + 1

    print(f"\n📂 Operation Categories:")
    for category, count in sorted(categories.items()):
        print(f"   {category}: {count} operations")


def list_all_operations(operations: Dict[str, Any]):
    """List all available operations"""
    print("\n📋 ALL AVAILABLE OPERATIONS")
    print("=" * 80)

    # Group by category
    categories = {
        "core": [],
        "experimental": [],
        "transformer": [],
        "moreh": [],
        "kv_cache": [],
        "prim": [],
        "other": [],
    }

    for op_name in sorted(operations.keys()):
        config_count = len(operations[op_name].get("configurations", []))
        op_display = f"{op_name} ({config_count} configs)"

        if "::experimental::" in op_name:
            categories["experimental"].append(op_display)
        elif "::transformer::" in op_name:
            categories["transformer"].append(op_display)
        elif "::moreh::" in op_name:
            categories["moreh"].append(op_display)
        elif "::kv_cache::" in op_name:
            categories["kv_cache"].append(op_display)
        elif "::prim::" in op_name:
            categories["prim"].append(op_display)
        elif op_name.startswith("ttnn::"):
            categories["core"].append(op_display)
        else:
            categories["other"].append(op_display)

    for category, ops in categories.items():
        if ops:
            print(f"\n🔧 {category.upper()} OPERATIONS ({len(ops)}):")
            for op in ops:
                print(f"   • {op}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TTNN operations master file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_operations.py add                    # Show ttnn::add configurations
  python analyze_operations.py linear --brief         # Show ttnn::linear without details
  python analyze_operations.py experimental.add       # Show ttnn::experimental::add
  python analyze_operations.py --list                 # List all operations
  python analyze_operations.py --stats                # Show statistics
        """,
    )

    parser.add_argument("operation", nargs="?", help="Operation name to search for")
    parser.add_argument("--list", action="store_true", help="List all available operations")
    parser.add_argument("--stats", action="store_true", help="Show master file statistics")
    parser.add_argument("--brief", action="store_true", help="Show brief output without configuration details")
    parser.add_argument("--debug", action="store_true", help="Show all arguments including nullopt (debug mode)")
    parser.add_argument(
        "--master-file",
        default="./model_tracer/traced_operations/ttnn_operations_master.json",
        help="Path to master JSON file (default: ./model_tracer/traced_operations/ttnn_operations_master.json)",
    )

    args = parser.parse_args()

    # Load master file
    print(f"📂 Loading master file: {args.master_file}")
    data = load_master_file(args.master_file)
    operations = data["operations"]

    print(f"✅ Loaded {len(operations)} operations")

    # Handle different modes
    if args.stats:
        show_statistics(data)
    elif args.list:
        list_all_operations(operations)
    elif args.operation:
        # Search for specific operation
        matches = find_matching_operations(operations, args.operation)

        if not matches:
            print(f"❌ No operations found matching '{args.operation}'")
            print("\n💡 Available operations:")
            all_ops = sorted(operations.keys())
            for op in all_ops[:10]:
                print(f"   • {op}")
            if len(all_ops) > 10:
                print(f"   ... and {len(all_ops) - 10} more (use --list to see all)")
        else:
            for match in matches:
                configurations = operations[match].get("configurations", [])
                print_operation_configs(match, configurations, not args.brief, args.debug)

                if len(matches) > 1 and match != matches[-1]:
                    print("\n" + "=" * 80)
    else:
        # No arguments provided, show help and basic info
        print(f"🎯 TTNN Operations Analyzer")
        print(f"📊 Master file contains {len(operations)} operations")
        print(f"💡 Use --help to see usage options")
        print(f"🔍 Quick examples:")
        print(f"   python analyze_operations.py add")
        print(f"   python analyze_operations.py --list")
        print(f"   python analyze_operations.py --stats")


if __name__ == "__main__":
    main()
