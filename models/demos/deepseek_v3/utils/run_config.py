# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, Optional, Union

import ttnn


def create_run_config(
    model_config: Dict[str, Any],
    weights_path: Union[str, Path],
    mesh_device: ttnn.Device,
    layer_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a runtime configuration by combining model config with loaded TTNN weights.

    This function:
    - Expands range patterns for specific layers (e.g., "model.layers.0-31.mlp" -> "model.layers.5.mlp")
    - Loads TTNN tensors from disk and adds them to the config
    - Returns a nested dict structure for easy access

    Args:
        model_config: Nested dict containing operator configs with TTNN objects
        weights_path: Path to directory containing TTNN weight files
        mesh_device: TTNN mesh device for loading weights
        layer_num: Optional layer number for range expansion

    Returns:
        Nested dict with structure matching model_config, plus loaded weights
    """
    weights_path = Path(weights_path)

    # If layer_num is provided, expand ranges first
    if layer_num is not None:
        model_config = _expand_ranges(model_config, layer_num)

    # Load weights into the config
    return _load_weights(model_config, weights_path, mesh_device)


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


def _load_weights(data: Any, weights_path: Path, mesh_device: ttnn.Device, current_key: str = "") -> Any:
    """Recursively load weights into the config structure.

    Args:
        data: The data to process (dict, list, or other)
        weights_path: Path to weights directory
        mesh_device: TTNN device for loading weights
        current_key: Current key path for weight loading (e.g., "w1", "attention.wq")

    Returns:
        Config structure with loaded weights added
    """
    if isinstance(data, dict):
        # Check if this dict represents an operation that might have weights
        weight_file = weights_path / f"{current_key}.weight" if current_key else None
        if weight_file and weight_file.exists():
            # Load the weight and add it to the dict
            data = data.copy()
            data["weight"] = ttnn.load_tensor(str(weight_file))

        # Process all nested dicts recursively
        result = {}
        for key, value in data.items():
            # Build the key path for nested structures
            new_key = f"{current_key}.{key}" if current_key else key
            result[key] = _load_weights(value, weights_path, mesh_device, new_key)

        return result

    elif isinstance(data, list):
        # Recursively process lists
        return [_load_weights(item, weights_path, mesh_device, current_key) for item in data]

    else:
        # Return other types as-is (including TTNN objects)
        return data
