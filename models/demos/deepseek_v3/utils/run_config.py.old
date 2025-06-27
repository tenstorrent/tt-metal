# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
from types import NoneType
from typing import Any, Optional, Union

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import ModelConfig, RunConfig, WeightsConfig
from models.demos.deepseek_v3.utils.config_dataclass import OpConfigBase


def create_run_config(
    model_config: ModelConfig,
    weight_config: WeightsConfig,
    mesh_device: ttnn.Device,
    layer_num: Optional[int] = None,
) -> RunConfig:
    """Create a runtime configuration by combining model config with loaded TTNN weights.

    This function:
    - Expands range patterns for specific layers (e.g., ["model"]["layers"]["0-31"]["mlp"] gets expanded to 32 layers)
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

    # Load weights from the weight config
    loaded_weight_config = _load_weight_config(weight_config, mesh_device)

    # Recursively traverse both configs together and
    result: RunConfig = _merge_model_weight_config(model_config, loaded_weight_config)
    print(pretty_print_run_config(result))
    return result


def pretty_print_run_config(run_config: dict[str, Any], indent: int = 0) -> str:
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
        elif dataclasses.is_dataclass(value):
            # Dataclass - format as nested structure showing all fields
            lines.append(f"{indent_str}{key}: {value.__class__.__name__}")
            # Get field values without deep copying (preserves ttnn object references)
            dataclass_dict = {}
            for field in dataclasses.fields(value):
                dataclass_dict[field.name] = getattr(value, field.name)
            lines.append(pretty_print_run_config(dataclass_dict, indent + 1))
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


def _format_ttnn_object(value: Any) -> Union[dict[str, Any], str]:
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


def _format_memory_config(memory_config) -> dict[str, Any]:
    """Format MemoryConfig object."""
    return {
        "memory_layout": str(memory_config.memory_layout).split("::")[-1],
        "buffer_type": str(memory_config.buffer_type).split("::")[-1],
        "shard_spec": _format_ttnn_object(memory_config.shard_spec) if memory_config.shard_spec else None,
        "nd_shard_spec": _format_ttnn_object(memory_config.nd_shard_spec) if memory_config.nd_shard_spec else None,
    }


def _format_shape(shape) -> dict[str, Any]:
    """Format Shape object."""
    return {
        "dimensions": list(shape),
        "rank": len(shape),
    }


def _format_data_type(data_type) -> str:
    """Format DataType object."""
    return str(data_type).split(".")[-1]


def _format_matmul_program_config(class_name, config) -> dict[str, Any]:
    """Format MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig object."""
    return {
        "type": class_name,
        "in0_block_w": config.in0_block_w,
        "per_core_M": config.per_core_M,
        "per_core_N": config.per_core_N,
        "fused_activation": _format_ttnn_object(config.fused_activation) if config.fused_activation else None,
    }


def _format_compute_kernel_config(config) -> dict[str, Any]:
    """Format WormholeComputeKernelConfig object."""
    return {
        "fp32_dest_acc_en": getattr(config, "fp32_dest_acc_en", None),
        "math_approx_mode": getattr(config, "math_approx_mode", None),
        "packer_l1_acc": getattr(config, "packer_l1_acc", None),
    }


def _format_shard_spec(shard_spec) -> dict[str, Any]:
    """Format ShardSpec object."""
    return {
        "grid": str(shard_spec.grid),
        "shape": _format_ttnn_object(shard_spec.shape) if hasattr(shard_spec, "shape") else None,
        "orientation": str(shard_spec.orientation).split("::")[-1],
        "mode": str(shard_spec.mode).split("::")[-1],
    }


def _format_nd_shard_spec(nd_shard_spec) -> dict[str, Any]:
    """Format NdShardSpec object."""
    return {
        "shard_shape": (
            _format_ttnn_object(nd_shard_spec.shard_shape) if hasattr(nd_shard_spec, "shard_shape") else None
        ),
        "grid": str(nd_shard_spec.grid),
        "orientation": str(nd_shard_spec.orientation).split("::")[-1],
    }


def _format_unary_op_type(op_type) -> dict[str, Any]:
    """Format UnaryOpType object."""
    return {
        "operation": str(op_type).split(".")[-1],
    }


def _format_generic_ttnn_object(obj) -> dict[str, Any]:
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
    # Check if this is a dataclass
    if dataclasses.is_dataclass(value):
        # Format dataclass as nested structure
        # Get field values without deep copying (preserves ttnn object references)
        dataclass_dict = {}
        for field in dataclasses.fields(value):
            dataclass_dict[field.name] = getattr(value, field.name)
        return f"{value.__class__.__name__}\n" + pretty_print_run_config(dataclass_dict, indent + 1)

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


def _expand_ranges(config: dict[str, Any], layer_num: int) -> dict[str, Any]:
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


def _merge_model_weight_config(model_value: Any, loaded_weight_value: Any) -> Any:
    """Recursively merge model_config and weight_config.

    Args:
        model_config: The model configuration (element)
        loaded_weight_value: The loaded weight configuration (element)

    Returns:
        Merged run config
    """

    if _is_op_config(model_value):
        op_config_dataclass = model_value.__class__
        op_config_dict = {
            field.name: getattr(model_value, field.name) for field in dataclasses.fields(op_config_dataclass)
        }
        return model_value.__class__(**_merge_model_weight_config(op_config_dict, loaded_weight_value))

    if isinstance(model_value, (dict, NoneType)) and isinstance(loaded_weight_value, (dict, NoneType)):
        if model_value is None and loaded_weight_value is None:
            return None
        model_value = model_value or {}
        loaded_weight_value = loaded_weight_value or {}
        return {
            k: _merge_model_weight_config(model_value.get(k, None), loaded_weight_value.get(k, None))
            for k in itertools.chain(model_value.keys(), loaded_weight_value.keys())
        }

    if model_value is None:
        return loaded_weight_value

    if loaded_weight_value is None:
        return model_value

    if isinstance(model_value, list) and isinstance(loaded_weight_value, list):
        if len(model_value) != len(loaded_weight_value):
            raise ValueError(
                f"Cannot merge config lists of different lengths: {len(model_value)} vs {len(loaded_weight_value)}"
            )
        return [_merge_model_weight_config(m, w) for m, w in zip(model_value, loaded_weight_value)]

    raise ValueError(f"Cannot merge {model_value} and {loaded_weight_value} config values")


def _load_weight_config(weight_value: Any, mesh_device: ttnn.Device) -> Any:
    """Load weights from the weight config (element).

    Args:
        weight_value: Either a weight (sub)config, or an element of thereof (to be loaded)
        mesh_device: TTNN device for loading weights

    Returns:
        Dictionary with loaded TTNN tensors
    """

    if isinstance(weight_value, str):
        return ttnn.load_tensor(weight_value, device=mesh_device)
    elif isinstance(weight_value, dict):
        return {key: _load_weight_config(value, mesh_device) for key, value in weight_value.items()}
    else:
        return weight_value


def _is_op_config(value: Any) -> bool:
    """Check if the value is an operator configuration (e.g., LinearConfig, EmbeddingConfig).

    Args:
        value: The value to check

    Returns:
        True if the value is an operator configuration, False otherwise
    """
    return issubclass(type(value), OpConfigBase)
