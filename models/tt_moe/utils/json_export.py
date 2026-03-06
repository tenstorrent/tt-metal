# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""JSON export utilities for MoE configurations.

This module provides utilities to serialize TTNN objects and data structures
to JSON-compatible formats for configuration export.
"""

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Union

from loguru import logger

import ttnn


def serialize_ttnn_object(obj: Any) -> Any:
    """Convert a single TTNN object to JSON-serializable format.

    Args:
        obj: TTNN object or primitive type

    Returns:
        JSON-serializable representation of the object
    """
    # Handle None
    if obj is None:
        return None

    # Handle primitive types
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # Skip WormholeComputeKernelConfig objects entirely
    if hasattr(obj, "__class__") and "ComputeKernelConfig" in obj.__class__.__name__:
        return None  # Will be filtered out by dictionary processing

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            serialized_item = serialize_ttnn_object(item)
            if serialized_item is not None:
                result.append(serialized_item)
        return result

    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Skip compute_kernel_config (hardware-specific constant)
            if key == "compute_kernel_config":
                continue
            serialized_value = serialize_ttnn_object(value)
            # Skip None values (filtered objects)
            if serialized_value is not None:
                result[key] = serialized_value
        return result

    # Handle dataclasses
    if is_dataclass(obj):
        # Manual dataclass conversion to avoid deepcopy issues
        result = {}
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            serialized_value = serialize_ttnn_object(value)
            # Skip None values to keep JSON cleaner
            if serialized_value is not None:
                result[field] = serialized_value
        return result

    # Handle TTNN memory configs (including ShardedMemoryConfig)
    obj_str = str(obj)
    if "MemoryConfig" in obj_str or "memory_config" in obj_str.lower():
        if "L1" in obj_str:
            return "L1"
        elif "DRAM" in obj_str:
            return "DRAM"
        elif "ShardedMemoryConfig" in obj_str:
            # Extract shard details if possible
            return {"type": "ShardedMemoryConfig", "details": obj_str}  # Keep string representation for now

    if hasattr(obj, "__name__"):
        obj_name = obj.__name__ if hasattr(obj, "__name__") else str(obj)
        if "L1_MEMORY_CONFIG" in obj_name or obj == ttnn.L1_MEMORY_CONFIG:
            return "L1"
        elif "DRAM_MEMORY_CONFIG" in obj_name or obj == ttnn.DRAM_MEMORY_CONFIG:
            return "DRAM"

    # Handle TTNN DataType
    if hasattr(ttnn, "DataType") and isinstance(obj, ttnn.DataType):
        return str(obj).split(".")[-1]  # e.g., "bfloat16", "bfloat8_b"

    # Handle TTNN Shape
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "Shape":
        # ttnn.Shape can be accessed as a list
        return list(obj)

    # Handle TTNN Topology
    if hasattr(ttnn, "Topology") and hasattr(obj, "__class__"):
        if "Topology" in str(obj.__class__):
            return str(obj).split(".")[-1]  # e.g., "Ring", "Linear"

    # Handle MeshDeviceStub
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "MeshDeviceStub":
        if hasattr(obj, "shape"):
            return {"mesh_shape": list(obj.shape)}
        elif hasattr(obj, "mesh_shape"):
            return {"mesh_shape": list(obj.mesh_shape)}

    # Handle torch tensors (convert to list for small tensors, skip for large)
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "numpy"):
        # Skip large tensors (e.g., weights)
        total_elements = 1
        for dim in obj.shape:
            total_elements *= dim
        if total_elements > 1000:
            return f"<Tensor shape={list(obj.shape)} dtype={str(obj.dtype)}>"
        else:
            return obj.tolist()

    # Handle classes/types by returning their name
    if isinstance(obj, type):
        return obj.__name__

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        # Try to extract relevant attributes
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):  # Skip private attributes
                result[key] = serialize_ttnn_object(value)
        if result:
            result["__type__"] = obj.__class__.__name__
            return result

    # Default: convert to string
    return str(obj)


def serialize_ttnn_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert TTNN configuration objects to JSON-serializable format.

    Args:
        config: Configuration dictionary containing TTNN objects

    Returns:
        JSON-serializable configuration dictionary
    """
    serialized = {}

    for key, value in config.items():
        # Skip certain runtime-only keys
        if key in ["ccl", "ccl_manager", "mesh_device", "device"]:
            logger.debug(f"Skipping runtime-only key: {key}")
            continue

        # Skip state dicts (too large for JSON)
        if key.endswith("_state_dict") or key == "state_dict":
            logger.debug(f"Skipping state dict: {key}")
            serialized[key] = "<state_dict>"
            continue

        # Skip compute_kernel_config (hardware-specific constant, set at runtime)
        if key == "compute_kernel_config":
            logger.debug(f"Skipping hardware-specific compute_kernel_config")
            continue

        # Recursively serialize the value
        serialized[key] = serialize_ttnn_object(value)

    return serialized


def export_config_to_json(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Export configuration dictionary to JSON file.

    Args:
        config: Configuration dictionary to export
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize the configuration
    serialized = serialize_ttnn_config(config)

    # Write to JSON file with pretty formatting
    with open(output_path, "w") as f:
        json.dump(serialized, f, indent=2, sort_keys=False)

    logger.info(f"Configuration exported to {output_path}")


def load_config_from_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Note: This loads the JSON but does not reconstruct TTNN objects.
    The loaded config will need to be processed to recreate TTNN objects.

    Args:
        json_path: Path to JSON file

    Returns:
        Configuration dictionary
    """
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        config = json.load(f)

    logger.info(f"Configuration loaded from {json_path}")

    # Process any placeholders in the config
    config = process_config_placeholders(config)

    return config


def process_config_placeholders(obj: Any) -> Any:
    """Process configuration placeholders and generate actual objects.

    For example, convert "<random tensor shape=[128, 2880]>" to an actual random tensor.

    Args:
        obj: Configuration object that may contain placeholders

    Returns:
        Configuration with placeholders replaced by actual objects
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        # Check for random tensor placeholders
        if obj.startswith("<random tensor shape="):
            import re

            import torch

            # Extract shape from string like "<random tensor shape=[128, 2880]>"
            match = re.match(r"<random tensor shape=\[([^\]]+)\]>", obj)
            if match:
                shape_str = match.group(1)
                shape = [int(x.strip()) for x in shape_str.split(",")]
                return torch.randn(*shape)
        return obj

    if isinstance(obj, dict):
        return {key: process_config_placeholders(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [process_config_placeholders(item) for item in obj]

    return obj
