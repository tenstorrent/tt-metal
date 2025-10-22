#!/usr/bin/env python3

import re
import sys
import pandas as pd
from collections import defaultdict

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("⚠️  tabulate not found. Install with: pip install tabulate")
    print("   Falling back to basic table formatting\n")

try:
    from colorama import Fore, Style, init

    init(autoreset=True)  # Auto-reset colors after each print
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

    # Fallback ANSI codes for basic color support
    class Fore:
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        MAGENTA = "\033[35m"  # Pink/Magenta
        RESET = "\033[0m"

    class Style:
        RESET_ALL = "\033[0m"


def get_status_indicator(status_type):
    """Get colored single-character status indicator."""
    if status_type == "match":
        return f"{Fore.RESET}o{Style.RESET_ALL}"
    elif status_type == "different":
        return f"{Fore.MAGENTA}\033[1mX\033[0m{Style.RESET_ALL}"  # Bold pink X
    elif status_type == "weight_dtype_mismatch":
        return f"{Fore.YELLOW}W{Style.RESET_ALL}"  # Yellow W for weight dtype
    elif status_type == "missing_in_unit":
        return f"{Fore.RED}U{Style.RESET_ALL}"  # Red U for missing in unit tests
    elif status_type == "missing_in_model":
        return f"{Fore.RED}M{Style.RESET_ALL}"  # Red M for missing in model
    else:
        return "?"


def is_numeric_value(value):
    """Check if a value is numeric (int, float, or numeric string)."""
    if value in ["MISSING", "None", "std::nullopt", "N/A", "DISABLED"]:
        return False

    # Try to convert to number
    str_val = str(value).strip()

    # Handle tuples like (1, 1) or (2, 2, 2, 2)
    if str_val.startswith("(") and str_val.endswith(")"):
        # Extract numbers from tuple
        import re

        numbers = re.findall(r"-?\d+(?:\.\d+)?", str_val)
        return len(numbers) > 0

    # Handle single numbers
    try:
        float(str_val)
        return True
    except (ValueError, TypeError):
        return False


def is_special_displayable(param_name):
    """Check if a parameter should be displayed even if not numeric (special handling)."""
    special_params = {"conv_config.shard_layout", "shard_layout", "slice_config"}
    return param_name in special_params


def format_shard_layout_value(value):
    """Format shard_layout values with custom mapping."""
    str_val = str(value).strip().lower()

    if "height" in str_val or "h_shard" in str_val:
        return "HS"
    elif "width" in str_val or "w_shard" in str_val:
        return "WS"
    elif "block" in str_val or "b_shard" in str_val:
        return "BS"
    elif str_val in ["none", "missing", "std::nullopt", "n/a"]:
        return "N/A"
    else:
        # If we can't categorize it, return a shortened version
        return str_val[:4].upper() if str_val else "N/A"


def format_slice_config_value(value):
    """Format slice_config SliceType values with custom mapping."""
    str_val = str(value).strip()

    if "SliceType::L1_FULL" in str_val or "L1_FULL" in str_val:
        return "L1"
    elif "SliceType::DRAM_WIDTH" in str_val or "DRAM_WIDTH" in str_val:
        return "DW"
    elif "SliceType::DRAM_HEIGHT" in str_val or "DRAM_HEIGHT" in str_val:
        return "DH"
    elif str_val.lower() in ["none", "missing", "std::nullopt", "n/a"]:
        return "N/A"
    else:
        # If we can't categorize it, return a shortened version
        return str_val[:4].upper() if str_val else "N/A"


def format_value_with_color(value, unit_value, model_value, param_name, is_unit_row=True):
    """Format a numeric or special value with color based on comparison status."""
    # Handle special formatting for specific parameters
    if param_name in ["conv_config.shard_layout", "shard_layout"]:
        formatted_value = format_shard_layout_value(value)
        formatted_unit = format_shard_layout_value(unit_value)
        formatted_model = format_shard_layout_value(model_value)
    elif param_name == "slice_config":
        formatted_value = format_slice_config_value(value)
        formatted_unit = format_slice_config_value(unit_value)
        formatted_model = format_slice_config_value(model_value)
    else:
        formatted_value = value
        formatted_unit = unit_value
        formatted_model = model_value

    if formatted_value == "MISSING" or formatted_value == "N/A":
        return f"{Fore.RED}-{Style.RESET_ALL}"

    # Normalize values for comparison
    unit_val_normalized = "MISSING" if formatted_unit in ["None", "MISSING"] else formatted_unit
    model_val_normalized = "MISSING" if formatted_model in ["None", "MISSING"] else formatted_model

    # Determine status
    if unit_val_normalized == "MISSING" and model_val_normalized == "MISSING":
        return f"{Fore.RED}-{Style.RESET_ALL}"
    elif unit_val_normalized == "MISSING":
        return f"{Fore.RED}-{Style.RESET_ALL}" if is_unit_row else f"{Fore.RED}{formatted_value}{Style.RESET_ALL}"
    elif model_val_normalized == "MISSING":
        return f"{Fore.RED}{formatted_value}{Style.RESET_ALL}" if is_unit_row else f"{Fore.RED}-{Style.RESET_ALL}"
    elif unit_val_normalized == model_val_normalized:
        return f"{Fore.RESET}{formatted_value}{Style.RESET_ALL}"  # Match - normal color
    else:
        # Check if it's specifically a weight tensor dtype mismatch
        if param_name == "weight_tensor_dtype":
            return f"{Fore.YELLOW}{formatted_value}{Style.RESET_ALL}"  # Yellow for weight dtype
        else:
            return f"{Fore.MAGENTA}\033[1m{formatted_value}\033[0m{Style.RESET_ALL}"  # Bold pink for differences

    return str(formatted_value)


def parse_conv2d_entry(line):
    """Parse a CONV2D_ARGS log line and extract layer name and parameters."""
    # Extract the layer name and parameters
    match = re.search(r"CONV2D_ARGS:([^:]+):(.*)", line)
    if not match:
        return None, {}

    layer_name = match.group(1)
    params_str = match.group(2)

    # Parse parameters - this is tricky because of nested objects
    params = {}

    # Split by comma, but be careful with nested objects
    param_parts = []
    current_part = ""
    paren_depth = 0
    angle_depth = 0

    for char in params_str:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "," and paren_depth == 0 and angle_depth == 0:
            param_parts.append(current_part.strip())
            current_part = ""
            continue

        current_part += char

    if current_part.strip():
        param_parts.append(current_part.strip())

    # Parse each parameter
    for param in param_parts:
        if "=" in param:
            key, value = param.split("=", 1)
            params[key.strip()] = value.strip()

    return layer_name, params


def normalize_value(value, param_name=None):
    """Normalize values for comparison."""
    # Remove object memory addresses
    value = re.sub(r"object at 0x[0-9a-f]+", "object", value)
    # Normalize DataType formats
    value = re.sub(r"DataType\.([A-Z0-9_]+)", r"DataType::\1", value)
    # Normalize None values
    if value.lower() in ["none", "std::nullopt"]:
        value = "None"

    # Special parameter-specific normalizations
    if param_name == "enable_kernel_stride_folding":
        # Treat 0 and None as equivalent (both mean disabled)
        if value in ["0", "None"]:
            return "DISABLED"
    elif param_name == "act_block_h_override":
        # Treat 0 and MISSING as equivalent (both mean no override)
        if value in ["0", "MISSING"]:
            return "N/A"
    elif param_name == "padding":
        # Normalize padding formats: (1, 1, 1, 1) should equal (1, 1) when they represent same padding
        # Pattern: (top, bottom, left, right) vs (vertical, horizontal)
        if value.startswith("(") and value.endswith(")"):
            # Extract numbers from padding tuple
            numbers = re.findall(r"\d+", value)
            if len(numbers) == 4:
                # (top, bottom, left, right) format
                top, bottom, left, right = numbers
                if top == bottom and left == right:
                    # Convert to (vertical, horizontal) format
                    return f"({top}, {left})"
            elif len(numbers) == 2:
                # (vertical, horizontal) format - expand to full format for comparison
                v, h = numbers
                return f"({v}, {h})"

    return value


def parse_conv_config(conv_config_str):
    """Parse Conv2dConfig into individual fields."""
    if not conv_config_str.startswith("Conv2dConfig("):
        return {}

    # Extract the content inside Conv2dConfig(...)
    content = conv_config_str[13:-1]  # Remove 'Conv2dConfig(' and ')'

    # Split by comma but be careful with nested objects
    params = []
    current_param = ""
    paren_depth = 0
    angle_depth = 0

    for char in content:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "," and paren_depth == 0 and angle_depth == 0:
            params.append(current_param.strip())
            current_param = ""
            continue
        current_param += char

    if current_param.strip():
        params.append(current_param.strip())

    # Parse into key-value pairs
    config_fields = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)
            config_fields[key.strip()] = value.strip()

    return config_fields


def format_conv_config(conv_config_str):
    """Format Conv2dConfig for better readability."""
    if not conv_config_str.startswith("Conv2dConfig("):
        return conv_config_str

    # Extract the content inside Conv2dConfig(...)
    content = conv_config_str[13:-1]  # Remove 'Conv2dConfig(' and ')'

    # Split by comma but be careful with nested objects
    params = []
    current_param = ""
    paren_depth = 0
    angle_depth = 0

    for char in content:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "," and paren_depth == 0 and angle_depth == 0:
            params.append(current_param.strip())
            current_param = ""
            continue
        current_param += char

    if current_param.strip():
        params.append(current_param.strip())

    # Format as multi-line with indentation
    formatted_params = []
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)
            formatted_params.append(f"    {key.strip()}={value.strip()}")
        else:
            formatted_params.append(f"    {param}")

    return "Conv2dConfig(\n" + ",\n".join(formatted_params) + "\n  )"


def compare_parameters(unit_params, model_params, layer_name, ignore_weight_dtype=False):
    """Compare parameters between unit test and model."""
    differences = []
    all_keys = set(unit_params.keys()) | set(model_params.keys())

    # Parameters to ignore during comparison
    ignored_params = {"return_output_dim", "return_weights_and_bias", "device", "weight_tensor_dtype"}

    # Note: weight_tensor_dtype and device are now always ignored

    all_keys = all_keys - ignored_params

    for key in sorted(all_keys):
        unit_val = normalize_value(unit_params.get(key, "MISSING"), key)
        model_val = normalize_value(model_params.get(key, "MISSING"), key)

        # Treat None and MISSING as equivalent
        if unit_val == "None":
            unit_val = "MISSING"
        if model_val == "None":
            model_val = "MISSING"

        if unit_val != model_val:
            differences.append({"param": key, "unit_test": unit_val, "model": model_val})

    return differences


def load_conv2d_data(filename):
    """Load and parse CONV2D_ARGS entries from a file."""
    data = defaultdict(list)

    try:
        with open(filename, "r") as f:
            for line_num, line in enumerate(f, 1):
                layer_name, params = parse_conv2d_entry(line.strip())
                if layer_name:
                    data[layer_name].append(params)
                else:
                    print(f"Warning: Could not parse line {line_num} in {filename}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return {}

    return data


def create_comparison_dataframe(unit_data, model_data):
    """Create a comparison table with two rows per layer: unit test data and model data."""
    # Get all unique layer names
    all_layers = set(unit_data.keys()) | set(model_data.keys())

    # Parameters to ignore during comparison
    ignored_params = {"return_output_dim", "return_weights_and_bias", "device", "weight_tensor_dtype"}
    # Note: weight_tensor_dtype and device are now always ignored

    # Collect all unique numeric parameters across all layers
    all_params = set()
    param_values_sample = {}  # Store sample values to check if they're numeric

    for layer in all_layers:
        if layer in unit_data and unit_data[layer]:
            unit_entry = unit_data[layer][0]
            for param, value in unit_entry.items():
                if param not in ignored_params:
                    if param == "conv_config":
                        # Expand conv_config fields
                        config_fields = parse_conv_config(value)
                        for field, field_value in config_fields.items():
                            param_name = f"conv_config.{field}"
                            normalized_val = normalize_value(field_value, field)
                            if is_numeric_value(normalized_val) or is_special_displayable(param_name):
                                all_params.add(param_name)
                                param_values_sample[param_name] = normalized_val
                    else:
                        normalized_val = normalize_value(value, param)
                        if is_numeric_value(normalized_val) or is_special_displayable(param):
                            all_params.add(param)
                            param_values_sample[param] = normalized_val

        if layer in model_data and model_data[layer]:
            model_entry = model_data[layer][0]
            for param, value in model_entry.items():
                if param not in ignored_params:
                    if param == "conv_config":
                        # Expand conv_config fields
                        config_fields = parse_conv_config(value)
                        for field, field_value in config_fields.items():
                            param_name = f"conv_config.{field}"
                            normalized_val = normalize_value(field_value, field)
                            if is_numeric_value(normalized_val) or is_special_displayable(param_name):
                                all_params.add(param_name)
                                param_values_sample[param_name] = normalized_val
                    else:
                        normalized_val = normalize_value(value, param)
                        if is_numeric_value(normalized_val) or is_special_displayable(param):
                            all_params.add(param)
                            param_values_sample[param] = normalized_val

    # Sort parameters for consistent ordering
    sorted_params = sorted(all_params)

    # Create table data structure (list of lists for better control)
    table_data = []
    headers = ["Layer/Source"] + sorted_params

    for layer in sorted(all_layers):
        # Get parameter values for this layer
        unit_params = {}
        model_params = {}

        if layer in unit_data and unit_data[layer]:
            unit_entry = unit_data[layer][0]
            for param, value in unit_entry.items():
                if param not in ignored_params:
                    if param == "conv_config":
                        config_fields = parse_conv_config(value)
                        for field, field_value in config_fields.items():
                            unit_params[f"conv_config.{field}"] = normalize_value(field_value, field)
                    else:
                        unit_params[param] = normalize_value(value, param)

        if layer in model_data and model_data[layer]:
            model_entry = model_data[layer][0]
            for param, value in model_entry.items():
                if param not in ignored_params:
                    if param == "conv_config":
                        config_fields = parse_conv_config(value)
                        for field, field_value in config_fields.items():
                            model_params[f"conv_config.{field}"] = normalize_value(field_value, field)
                    else:
                        model_params[param] = normalize_value(value, param)

        # Create unit test row (first row for this layer) - only for numeric params
        unit_row = [f"{layer[:12]}(U)"]  # Compact layer name with (U) for unit
        for param in sorted_params:
            unit_val = unit_params.get(param, "MISSING")
            model_val = model_params.get(param, "MISSING")
            # Only add if this parameter was identified as numeric
            if param in all_params:
                formatted_val = format_value_with_color(unit_val, unit_val, model_val, param, is_unit_row=True)
                unit_row.append(formatted_val)

        # Create model row (second row for this layer) - only for numeric params
        model_row = [f"{layer[:12]}(M)"]  # Compact layer name with (M) for model
        for param in sorted_params:
            unit_val = unit_params.get(param, "MISSING")
            model_val = model_params.get(param, "MISSING")
            # Only add if this parameter was identified as numeric
            if param in all_params:
                formatted_val = format_value_with_color(model_val, unit_val, model_val, param, is_unit_row=False)
                model_row.append(formatted_val)

        table_data.append(unit_row)
        table_data.append(model_row)

        # Add separator row after each unit/model pair (except for the last one)
        if layer != sorted(all_layers)[-1]:
            separator_row = ["─" * 12] + ["─"] * len(sorted_params)
            table_data.append(separator_row)

    return table_data, headers


def print_comparison_table(table_data, headers):
    """Print the comparison table using tabulate for clean formatting."""
    print("\n🔍 CONV2D Parameters Comparison")
    print("=" * 65)
    print(
        f"Legend: {Fore.RESET}Normal{Style.RESET_ALL}=Match | {Fore.MAGENTA}Pink{Style.RESET_ALL}=Diff | {Fore.RED}-{Style.RESET_ALL}=Missing | (U)=Unit (M)=Model"
    )
    print("ShL: HS=Height, WS=Width, BS=Block Sharded, N/A=None")
    print("SlC: L1=L1_FULL, DW=DRAM_WIDTH, DH=DRAM_HEIGHT, N/A=None")
    print("=" * 65)

    if not HAS_TABULATE:
        # Fallback to basic formatting if tabulate is not available
        print("Basic table format (install tabulate for better formatting):")
        for i, header in enumerate(headers):
            print(f"{header:>15}", end=" ")
        print()
        print("-" * (len(headers) * 16))
        for row in table_data:
            for i, value in enumerate(row):
                if i == 0:  # Layer column
                    print(f"{str(value):>15}", end=" ")
                else:  # Emoji columns
                    print(f"{str(value):>15}", end=" ")
            print()
        return

    # Custom column ordering and renaming
    column_order_mapping = {
        # Basic parameters (keep these in order)
        "batch_size": "N",
        "in_channels": "Cin",
        "out_channels": "Co",
        "input_height": "H",
        "input_width": "W",
        # Kernel parameters
        "kernel_size": "K",
        "padding": "P",
        "stride": "S",
        "groups": "G",
        "dilation": "D",
        # Data types and tensors
        "bias_tensor": "B?",
        "dtype": "D_T",
        "input_tensor_dtype": "D_I",
        # Memory and configuration
        "memory_config": "mem",
        "slice_config": "SlC",
        "compute_config": "CC",
        "shard_layout": "ShL_T",  # Top-level shard_layout
        # Conv config parameters
        "conv_config.act_block_h_override": "A_H",
        "conv_config.act_block_w_div": "A_W",
        "conv_config.activation": "A",
        "conv_config.config_tensors_in_dram": "C_T",
        "conv_config.core_grid": "C_G",
        "conv_config.deallocate_activation": "D_A",
        "conv_config.enable_act_double_buffer": "E_A",
        "conv_config.enable_activation_reuse": "E_R",
        "conv_config.enable_kernel_stride_folding": "E_K",
        "conv_config.enable_weights_double_buffer": "E_W",
        "conv_config.force_split_reader": "F_S",
        "conv_config.full_inner_dim": "F_I",
        "conv_config.in_place": "I_P",
        "conv_config.output_layout": "O_L",
        "conv_config.override_sharding_config": "O_S",
        "conv_config.reallocate_halo_output": "R_H",
        "conv_config.reshard_if_not_optimal": "R_O",
        "conv_config.shard_layout": "ShL",
        "conv_config.transpose_shards": "T_S",
        "conv_config.weights_dtype": "D_W",
    }

    # Columns to drop
    dropped_columns = {
        "act_block_h_override",  # Drop this top-level parameter
        "shard_layout",  # Hide top-level shard_layout (ShL_T)
        "weight_tensor_dtype",  # Ignore weight_tensor_dtype
        "device",  # Ignore device
    }

    # Create ordered headers based on the intended column_order_mapping sequence
    ordered_headers = ["Layer/Source"]  # Always start with Layer
    column_mapping = {}
    ordered_original_names = []

    # Define the intended column order
    intended_order = [
        "batch_size",
        "in_channels",
        "out_channels",
        "input_height",
        "input_width",
        "kernel_size",
        "padding",
        "stride",
        "groups",
        "dilation",
        "bias_tensor",
        "dtype",
        "input_tensor_dtype",
        "memory_config",
        "slice_config",
        "compute_config",
        "conv_config.shard_layout",
        "conv_config.act_block_h_override",
        "conv_config.act_block_w_div",
        "conv_config.activation",
        "conv_config.config_tensors_in_dram",
        "conv_config.core_grid",
        "conv_config.deallocate_activation",
        "conv_config.enable_act_double_buffer",
        "conv_config.enable_activation_reuse",
        "conv_config.enable_kernel_stride_folding",
        "conv_config.enable_weights_double_buffer",
        "conv_config.force_split_reader",
        "conv_config.full_inner_dim",
        "conv_config.in_place",
        "conv_config.output_layout",
        "conv_config.override_sharding_config",
        "conv_config.reallocate_halo_output",
        "conv_config.reshard_if_not_optimal",
        "conv_config.transpose_shards",
        "conv_config.weights_dtype",
    ]

    # Add columns in the intended order, but only if they exist in headers
    available_headers = set(headers[1:])  # Skip 'Layer/Source'
    for original_name in intended_order:
        if (
            original_name in available_headers
            and original_name not in dropped_columns
            and original_name in column_order_mapping
        ):
            short_name = column_order_mapping[original_name]
            ordered_headers.append(short_name)
            column_mapping[original_name] = short_name
            ordered_original_names.append(original_name)

    short_headers = ordered_headers

    # Filter and reorder table data to match new column ordering with separators
    filtered_table_data = []
    separator_positions = [6, 11, 14, 17]  # After W, D, input_tensor_dtype, compute_config

    for row in table_data:
        filtered_row = [row[0]]  # Keep layer name
        col_index = 1

        # Add columns in the intended order
        for original_header in ordered_original_names:
            # Find the index of this header in the original headers list
            if original_header in headers:
                original_index = headers.index(original_header)

                # Add separator before specific columns
                if col_index in separator_positions:
                    filtered_row.append("")  # Empty separator column

                filtered_row.append(row[original_index])
                col_index += 1

        filtered_table_data.append(filtered_row)

    # Add empty separator columns to headers
    grouped_headers = [short_headers[0]]  # Start with 'Layer'
    col_index = 1

    for i, header in enumerate(short_headers[1:], 1):
        # Add separator before specific columns
        if col_index in separator_positions:
            grouped_headers.append("  ")  # Empty separator header with min width

        grouped_headers.append(header)
        col_index += 1

    # Use tabulate to create the table with compact formatting
    table_output = tabulate(
        filtered_table_data,
        headers=grouped_headers,
        tablefmt="simple",  # Compact table without borders
        stralign="right",  # Right align for better numeric readability
        numalign="right",  # Right align numbers
    )

    print(table_output)

    # Print column legend for reference (only numeric parameters)
    print(f"\n📋 Column Legend ({len(column_mapping)} numeric parameters):")
    print("-" * 45)
    for original, short in column_mapping.items():
        print(f"{short:>4} = {original}")
    if not column_mapping:
        print("No numeric parameters found to display.")


def print_summary(table_data):
    """Print summary statistics."""
    # Filter out separator rows (rows that start with dashes or horizontal line chars)
    data_rows = [row for row in table_data if not str(row[0]).startswith(("-", "─"))]

    # Since we now have 2 data rows per layer, calculate actual layer count
    total_layers = len(data_rows) // 2

    # Count differences by analyzing pairs of rows (unit + model for each layer)
    match_count = 0
    diff_count = 0
    missing_count = 0

    # Process pairs of data rows (unit + model)
    for i in range(0, len(data_rows), 2):
        if i + 1 >= len(data_rows):
            break

        unit_row = data_rows[i]
        model_row = data_rows[i + 1]

        layer_has_diff = False
        layer_has_missing = False

        # Compare corresponding columns between unit and model rows
        for j in range(1, len(unit_row)):  # Skip first column (layer name)
            unit_val_str = str(unit_row[j]) if j < len(unit_row) else "MISSING"
            model_val_str = str(model_row[j]) if j < len(model_row) else "MISSING"

            # Look for color codes indicating differences or missing values
            if "MISSING" in unit_val_str or "MISSING" in model_val_str:
                layer_has_missing = True
            elif (
                "\033[35m" in unit_val_str
                or "\033[35m" in model_val_str
                or "\033[33m" in unit_val_str  # Magenta (different)
                or "\033[33m" in model_val_str
            ):  # Yellow (weight dtype)
                layer_has_diff = True

        if layer_has_missing:
            missing_count += 1
        elif layer_has_diff:
            diff_count += 1
        else:
            match_count += 1

    print(f"\n📈 SUMMARY:")
    print("=" * 50)
    print(f"✅ Layers with all parameters matching: {match_count}")
    print(f"❌ Layers with parameter differences: {diff_count}")
    print(f"⚠️  Layers with missing parameters: {missing_count}")
    print(f"📊 Total layers analyzed: {total_layers}")

    # Calculate overall health percentage
    health_pct = (match_count / total_layers * 100) if total_layers > 0 else 0
    print(f"🎯 Overall compatibility: {health_pct:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🔍 CONV2D_ARGS Log Comparison Tool - Compare numeric CONV2D parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_conv2d_logs.py                         # Display numeric parameter comparison

The script compares numeric CONV2D_ARGS entries between:
  - unit_tests_conv2d.txt (from unit test logs)
  - resnet_model_conv2d.txt (from ResNet model logs)

Each layer shows two compact rows: (U) for unit test, (M) for model data.
Only numeric values displayed with colors: Normal=Match, Pink=Different, Red(-)=Missing.
        """,
    )

    args = parser.parse_args()

    print("🔍 CONV2D_ARGS Numeric Parameters Comparison")
    print("📋 Mode: Compact Colored Value Display")
    print("=" * 50)

    # Load data from both files
    print("📁 Loading unit test data...")
    unit_data = load_conv2d_data("unit_tests_conv2d.txt")
    print(f"   Found {len(unit_data)} unique layers in unit tests")

    print("📁 Loading ResNet model data...")
    model_data = load_conv2d_data("resnet_model_conv2d.txt")
    print(f"   Found {len(model_data)} unique layers in ResNet model")

    if not unit_data and not model_data:
        print("❌ Error: Could not load data from either file")
        return 1

    # Create comparison table
    print("\n🔄 Processing comparison data...")
    table_data, headers = create_comparison_dataframe(unit_data, model_data)

    # Print the comparison table
    print_comparison_table(table_data, headers)

    # Print summary
    print_summary(table_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
