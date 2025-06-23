# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union

import ttnn


def create_run_config(
    model_config: Dict[str, Any],
    weight_config: Dict[str, Any],
    mesh_device: ttnn.Device,
    layer_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a runtime configuration by combining model config with loaded TTNN weights.

    This function:
    - Expands range patterns for specific layers (e.g., "model.layers.0-31.mlp" -> "model.layers.5.mlp")
    - Recursively traverses both weight_config and model_config together
    - Loads TTNN tensors where weight_config has entries and adds them to the model config
    - Returns a clean nested dict structure for easy access

    Args:
        model_config: Nested dict containing operator configs with TTNN objects
        weight_config: Nested dict mapping operation names to their TTNN weight file paths
        mesh_device: TTNN mesh device for loading weights
        layer_num: Optional layer number for range expansion

    Returns:
        Nested dict with structure matching model_config, plus loaded weights
    """
    # If layer_num is provided, expand ranges first
    if layer_num is not None:
        model_config = _expand_ranges(model_config, layer_num)
        weight_config = _expand_ranges(weight_config, layer_num)

    # Recursively traverse both configs together and load weights
    result = _merge_configs_and_load_weights(model_config, weight_config, mesh_device)
    print(pretty_print_run_config(result))
    return result


def pretty_print_run_config(run_config: Dict[str, Any], indent: int = 0) -> str:
    """Pretty print a run_config for human readability.

    Args:
        run_config: The run_config dict to print
        indent: Current indentation level (used internally for recursion)

    Returns:
        Formatted string representation of the run_config
    """
    lines = []
    indent_str = "  " * indent

    for key, value in sorted(run_config.items()):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            # TTNN tensor - treat as dict with shape, dtype, and memory_config children
            lines.append(f"{indent_str}{key}:")
            tensor_dict = {
                "shape": value.shape,
                "dtype": value.dtype,
                "memory_config": value.memory_config() if hasattr(value, "memory_config") else "None",
            }
            lines.append(pretty_print_run_config(tensor_dict, indent + 1))
        elif hasattr(value, "__class__") and value.__class__.__module__.startswith("ttnn"):
            # Other TTNN objects - format nicely
            ttnn_result = _format_ttnn_object(value)
            if ttnn_result is not None:
                if isinstance(ttnn_result, str):
                    # Simple string result (like DataType)
                    lines.append(f"{indent_str}{key}: {ttnn_result}")
                else:
                    # Dictionary result - format as nested structure
                    lines.append(f"{indent_str}{key}:")
                    lines.append(pretty_print_run_config(ttnn_result, indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {str(value)}")
        elif isinstance(value, dict):
            # Check if this dict has only one child
            if len(value) == 1:
                child_key, child_value = next(iter(value.items()))
                if isinstance(child_value, dict) and len(child_value) == 1:
                    # Multiple single children, format as nested
                    lines.append(f"{indent_str}{key}:")
                    lines.append(pretty_print_run_config(value, indent + 1))
                else:
                    # Single child, format on one line
                    child_str = _format_value(child_value, indent + 1)
                    lines.append(f"{indent_str}{key}.{child_key}: {child_str}")
            else:
                # Multiple children, format as nested
                lines.append(f"{indent_str}{key}:")
                lines.append(pretty_print_run_config(value, indent + 1))
        else:
            # Leaf value
            value_str = _format_value(value, indent)
            lines.append(f"{indent_str}{key}: {value_str}")

    return "\n".join(lines)


def _format_ttnn_object(value: Any) -> Union[Dict[str, Any], str]:
    """Format TTNN objects into readable dictionaries or strings.

    Args:
        value: The TTNN object to format

    Returns:
        Dictionary representation of the TTNN object, or string for simple types
    """
    if hasattr(value, "__class__") and value.__class__.__module__.startswith("ttnn"):
        class_name = value.__class__.__name__

        if class_name == "MemoryConfig":
            return _format_memory_config(value)
        elif class_name == "Shape":
            return _format_shape(value)
        elif class_name == "DataType":
            return _format_data_type(value)
        elif class_name == "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig":
            return _format_matmul_program_config(class_name, value)
        elif class_name == "WormholeComputeKernelConfig":
            return _format_compute_kernel_config(value)
        elif class_name == "ShardSpec":
            return _format_shard_spec(value)
        elif class_name == "NdShardSpec":
            return _format_nd_shard_spec(value)
        elif class_name == "UnaryOpType":
            return _format_unary_op_type(value)
        else:
            # For other TTNN objects, try to extract common attributes
            return _format_generic_ttnn_object(value)

    return None


def _format_memory_config(memory_config) -> Dict[str, Any]:
    """Format MemoryConfig object."""
    return {
        "memory_layout": str(memory_config.memory_layout).split("::")[-1],
        "buffer_type": str(memory_config.buffer_type).split("::")[-1],
        "shard_spec": _format_ttnn_object(memory_config.shard_spec) if memory_config.shard_spec else None,
        "nd_shard_spec": _format_ttnn_object(memory_config.nd_shard_spec) if memory_config.nd_shard_spec else None,
    }


def _format_shape(shape) -> Dict[str, Any]:
    """Format Shape object."""
    return {
        "dimensions": list(shape),
        "rank": len(shape),
    }


def _format_data_type(data_type) -> str:
    """Format DataType object."""
    return str(data_type).split(".")[-1]


def _format_matmul_program_config(class_name, config) -> Dict[str, Any]:
    """Format MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig object."""
    return {
        "type": class_name,
        "in0_block_w": config.in0_block_w,
        "per_core_M": config.per_core_M,
        "per_core_N": config.per_core_N,
        "fused_activation": _format_ttnn_object(config.fused_activation) if config.fused_activation else None,
    }


def _format_compute_kernel_config(config) -> Dict[str, Any]:
    """Format WormholeComputeKernelConfig object."""
    return {
        "fp32_dest_acc_en": getattr(config, "fp32_dest_acc_en", None),
        "math_approx_mode": getattr(config, "math_approx_mode", None),
        "packer_l1_acc": getattr(config, "packer_l1_acc", None),
    }


def _format_shard_spec(shard_spec) -> Dict[str, Any]:
    """Format ShardSpec object."""
    return {
        "grid": str(shard_spec.grid),
        "shape": _format_ttnn_object(shard_spec.shape) if hasattr(shard_spec, "shape") else None,
        "orientation": str(shard_spec.orientation).split("::")[-1],
        "mode": str(shard_spec.mode).split("::")[-1],
    }


def _format_nd_shard_spec(nd_shard_spec) -> Dict[str, Any]:
    """Format NdShardSpec object."""
    return {
        "shard_shape": _format_ttnn_object(nd_shard_spec.shard_shape)
        if hasattr(nd_shard_spec, "shard_shape")
        else None,
        "grid": str(nd_shard_spec.grid),
        "orientation": str(nd_shard_spec.orientation).split("::")[-1],
    }


def _format_unary_op_type(op_type) -> Dict[str, Any]:
    """Format UnaryOpType object."""
    return {
        "operation": str(op_type).split(".")[-1],
    }


def _format_generic_ttnn_object(obj) -> Dict[str, Any]:
    """Format generic TTNN objects by extracting common attributes."""
    result = {"type": obj.__class__.__name__}

    # Try to extract common attributes
    common_attrs = ["value", "name", "type", "config", "params"]
    for attr in common_attrs:
        if hasattr(obj, attr):
            try:
                result[attr] = getattr(obj, attr)
            except:
                pass

    return result


def _format_value(value: Any, indent: int) -> str:
    """Format a single value for pretty printing.

    Args:
        value: The value to format
        indent: Current indentation level

    Returns:
        Formatted string representation of the value
    """
    # Check if this is a TTNN object that can be formatted nicely
    ttnn_result = _format_ttnn_object(value)
    if ttnn_result is not None:
        if isinstance(ttnn_result, str):
            # Simple string result (like DataType)
            return ttnn_result
        else:
            # Dictionary result - format as nested structure
            return "\n" + pretty_print_run_config(ttnn_result, indent + 1)

    if isinstance(value, dict):
        # Nested dict
        if len(value) == 1:
            # Single child, format inline
            child_key, child_value = next(iter(value.items()))
            child_str = _format_value(child_value, indent)
            return f"{child_key}: {child_str}"
        else:
            # Multiple children, format as nested
            return "\n" + pretty_print_run_config(value, indent + 1)
    elif isinstance(value, list):
        # Check if this is a list of TTNN objects
        ttnn_objects = []
        other_objects = []
        for item in value:
            ttnn_result = _format_ttnn_object(item)
            if ttnn_result is not None:
                ttnn_objects.append((item, ttnn_result))
            else:
                other_objects.append(item)

        # If all items are TTNN objects, format them nicely
        if len(ttnn_objects) == len(value) and len(value) <= 3:
            if len(value) == 1:
                # Single item - just show the value directly
                ttnn_result = ttnn_objects[0][1]
                if isinstance(ttnn_result, str):
                    return ttnn_result
                else:
                    return ttnn_result.get("operation", ttnn_result.get("type", "TTNN_Object"))
            else:
                # Multiple items - format as multi-line
                lines = []
                for i, (item, ttnn_result) in enumerate(ttnn_objects):
                    if isinstance(ttnn_result, str):
                        item_str = ttnn_result
                    else:
                        item_str = ttnn_result.get("operation", ttnn_result.get("type", "TTNN_Object"))
                    if i == 0:
                        lines.append(f"[{item_str}")
                    else:
                        lines.append(f"  {item_str}")
                lines.append("]")
                return "\n".join(lines)

        # Regular list formatting
        if len(value) <= 3:
            items = []
            for item in value:
                # Check if this item is a TTNN object that can be formatted nicely
                ttnn_result = _format_ttnn_object(item)
                if ttnn_result is not None:
                    if isinstance(ttnn_result, str):
                        items.append(ttnn_result)
                    else:
                        # For lists, use a simpler format for TTNN objects
                        if "type" in ttnn_result and len(ttnn_result) == 1:
                            items.append(ttnn_result["type"])
                        else:
                            items.append(f"<{ttnn_result.get('type', 'TTNN_Object')}>")
                else:
                    items.append(_format_value(item, indent))
            return f"[{', '.join(items)}]"
        else:
            first_items = []
            for item in value[:2]:
                ttnn_result = _format_ttnn_object(item)
                if ttnn_result is not None:
                    if isinstance(ttnn_result, str):
                        first_items.append(ttnn_result)
                    else:
                        # For lists, use a simpler format for TTNN objects
                        if "type" in ttnn_result and len(ttnn_result) == 1:
                            first_items.append(ttnn_result["type"])
                        else:
                            first_items.append(f"<{ttnn_result.get('type', 'TTNN_Object')}>")
                else:
                    first_items.append(_format_value(item, indent))
            return f"[{', '.join(first_items)}, ... ({len(value)} total)]"
    else:
        # Regular value
        return str(value)


def _parse_range(range_str: str) -> tuple[int, int]:
    """Parse a range string like '0-31' or single number like '10' into (start, end) tuple."""
    if "-" in range_str:
        parts = range_str.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {range_str}. Expected format: 'start-end' or single number")

        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid range format: {range_str}. Range bounds must be integers")

        if start > end:
            raise ValueError(f"Invalid range: {range_str}. Start must be <= end")
    else:
        try:
            start = end = int(range_str)
        except ValueError:
            raise ValueError(f"Invalid layer number: {range_str}. Must be an integer")

    return start, end


def _expand_ranges(config: Dict[str, Any], layer_num: int) -> Dict[str, Any]:
    """Expand range patterns in model config for a specific layer.

    Handles patterns like:
    - "model.layers.0-31.mlp" -> "model.layers.5.mlp" (if layer_num=5)
    - "model.layers.10.attention" -> kept only if layer_num=10
    """
    # This is specifically for the model.layers.X pattern
    if "model" in config and isinstance(config["model"], dict):
        if "layers" in config["model"] and isinstance(config["model"]["layers"], dict):
            expanded_layers = {}
            seen_keys = {}  # Track overlaps

            for key, value in config["model"]["layers"].items():
                # Try to parse as range or single number
                try:
                    start, end = _parse_range(key)

                    # Check if current layer is in this range
                    if start <= layer_num <= end:
                        # Check for overlaps
                        for subkey in value:
                            if subkey in seen_keys:
                                raise ValueError(
                                    f"Overlapping ranges detected for key 'model.layers.{layer_num}.{subkey}': "
                                    f"ranges {seen_keys[subkey]} and {key} both define this key"
                                )
                            seen_keys[subkey] = key

                        # Merge this config into the layer config
                        if layer_num not in expanded_layers:
                            expanded_layers[layer_num] = {}
                        expanded_layers[layer_num].update(value)

                except ValueError:
                    # Not a range pattern, keep as-is
                    expanded_layers[key] = value

            # Replace layers with expanded version
            config = config.copy()
            config["model"] = config["model"].copy()
            config["model"]["layers"] = expanded_layers

    return config


def _merge_configs_and_load_weights(
    model_config: Dict[str, Any], weight_config: Dict[str, Any], mesh_device: ttnn.Device
) -> Dict[str, Any]:
    """Recursively merge model_config and weight_config, loading weights where weight_config has entries.

    Args:
        model_config: The model configuration dict
        weight_config: The weight configuration dict mapping to file paths
        mesh_device: TTNN device for loading weights

    Returns:
        Merged config with loaded weights added
    """
    if isinstance(model_config, dict) and isinstance(weight_config, dict):
        result = {}

        # Process all keys that exist in either config
        all_keys = set(model_config.keys()) | set(weight_config.keys())

        for key in all_keys:
            model_value = model_config.get(key, {})
            weight_value = weight_config.get(key, {})

            if isinstance(weight_value, dict) and weight_value:
                # This key has weight entries, load the weights
                if isinstance(model_value, dict):
                    # Merge model config with loaded weights
                    merged_value = model_value.copy()
                    for weight_key, weight_path in weight_value.items():
                        if isinstance(weight_path, str):
                            # Load the TTNN tensor from the file path
                            weight_tensor = ttnn.load_tensor(weight_path)
                            merged_value[weight_key] = weight_tensor
                        else:
                            # Keep non-string values as-is (e.g., nested configs)
                            merged_value[weight_key] = weight_path
                    result[key] = merged_value
                else:
                    # Model config doesn't have this key as a dict, create it with weights
                    merged_value = {}
                    for weight_key, weight_path in weight_value.items():
                        if isinstance(weight_path, str):
                            weight_tensor = ttnn.load_tensor(weight_path)
                            merged_value[weight_key] = weight_tensor
                        else:
                            merged_value[weight_key] = weight_path
                    result[key] = merged_value
            else:
                # No weight entries for this key, recursively process nested dicts
                if isinstance(model_value, dict) and isinstance(weight_value, dict):
                    result[key] = _merge_configs_and_load_weights(model_value, weight_value, mesh_device)
                else:
                    # Keep model value as-is
                    result[key] = model_value

        return result

    elif isinstance(model_config, list) and isinstance(weight_config, list):
        # Handle lists by processing each element
        return [_merge_configs_and_load_weights(m, w, mesh_device) for m, w in zip(model_config, weight_config)]

    else:
        # Return model_config as-is for non-dict/non-list types
        return model_config
