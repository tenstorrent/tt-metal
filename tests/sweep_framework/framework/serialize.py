# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ttnn import *
import json
import importlib
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


# Whitelist of allowed modules and types for secure deserialization
ALLOWED_MODULES = {
    "ttnn",
    "ttnn.types",
    "ttnn.tensor",
    "ttnn.operations",
    "ttnn.core",
    "ttnn.device",
    "ttnn.layout",
    "ttnn.memory_config",
    # Add other trusted modules as needed
}

ALLOWED_TYPES = {
    # Built-in safe types
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "tuple",
    # TTNN specific types that might appear without module prefix
    "DataType",
    "Layout",
    "MemoryConfig",
    "BufferType",
    "TensorMemoryLayout",
    "ShardSpec",
    "CoreRangeSet",
    "CoreRange",
    "CoreCoord",
    "ShardOrientation",
}


def safe_get_type(type_string):
    """
    Safely get a type from a string without using eval().
    Only allows whitelisted modules and types.
    """
    try:
        # Handle built-in types
        if type_string in ALLOWED_TYPES:
            return eval(type_string)  # Safe for built-in types

        # Handle module.ClassName format
        if "." in type_string:
            module_path, class_name = type_string.rsplit(".", 1)

            # Check if module is in whitelist
            root_module = module_path.split(".")[0]
            if root_module not in ALLOWED_MODULES:
                raise ValueError(f"Module {root_module} not in whitelist")

            # Import module and get class
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        else:
            # Single name, check if it's an allowed type
            if type_string in ALLOWED_TYPES:
                return eval(type_string)  # Safe for whitelisted types
            else:
                raise ValueError(f"Type {type_string} not in whitelist")
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to safely get type {type_string}: {e}")
        raise ValueError(f"Invalid or disallowed type: {type_string}")


def safe_eval_literal(value_string):
    """
    Safely evaluate string literals without arbitrary code execution.
    Only handles basic Python literals.
    """
    import ast

    try:
        # ast.literal_eval only evaluates literals, not arbitrary expressions
        return ast.literal_eval(value_string)
    except (ValueError, SyntaxError):
        # If it's not a valid literal, return as string
        return value_string


def convert_enum_values_to_strings(data):
    """Convert enum integer values to human-readable strings for PostgreSQL storage."""
    if not isinstance(data, dict):
        return data

    # Create a copy to avoid modifying the original
    result = data.copy()

    # Define enum mappings - from tt_metal/api/tt-metalium/buffer_types
    buffer_type_map = {0: "DRAM", 1: "L1", 2: "SYSTEM_MEMORY", 3: "L1_SMALL", 4: "TRACE"}

    memory_layout_map = {0: "INTERLEAVED", 2: "HEIGHT_SHARDED", 3: "WIDTH_SHARDED", 4: "BLOCK_SHARDED"}

    # Convert buffer_type if present
    if "buffer_type" in result and isinstance(result["buffer_type"], int):
        result["buffer_type"] = buffer_type_map.get(result["buffer_type"], f"UNKNOWN_{result['buffer_type']}")

    # Convert memory_layout if present
    if "memory_layout" in result and isinstance(result["memory_layout"], int):
        result["memory_layout"] = memory_layout_map.get(result["memory_layout"], f"UNKNOWN_{result['memory_layout']}")

    # Recursively process nested objects (like shard_spec)
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = convert_enum_values_to_strings(value)
        elif isinstance(value, list):
            result[key] = [convert_enum_values_to_strings(item) if isinstance(item, dict) else item for item in value]

    return result


def serialize(object, warnings=[]):
    if "to_json" in dir(object):
        return {"type": str(type(object)).split("'")[1], "data": object.to_json()}
    elif "pybind" in str(type(type(object))) and type(object) and type(object) not in warnings:
        logger.warning(
            f"pybinded ttnn class detected without a to_json method. Your type may need to pybind the to_json and from_json methods in C++, see the FAQ in the sweeps README for instructions. The type is {type(object)}. You can ignore this if this is an enum type."
        )
        warnings.append(type(object))
        return str(object)
    else:
        return str(object)


def deserialize(object):
    if isinstance(object, dict):
        type_cls = safe_get_type(object["type"])
        return type_cls.from_json(object["data"])
    else:
        try:
            return safe_eval_literal(object)
        except:
            return str(object)


def serialize_for_postgres(object, warnings=[]):
    if "to_json" in dir(object):
        json_str = object.to_json()
        try:
            # Parse the JSON string to make it queryable in PostgreSQL JSONB
            parsed_data = json.loads(json_str)
            # Convert enum integers to human-readable strings
            parsed_data = convert_enum_values_to_strings(parsed_data)
            return {"type": str(type(object)).split("'")[1], "data": parsed_data}
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, fall back to storing as string
            return {"type": str(type(object)).split("'")[1], "data": json_str}
    elif "pybind" in str(type(type(object))) and type(object) and type(object) not in warnings:
        logger.warning(
            f"pybinded ttnn class detected without a to_json method. Your type may need to pybind the to_json and from_json methods in C++, see the FAQ in the sweeps README for instructions. The type is {type(object)}. You can ignore this if this is an enum type."
        )
        warnings.append(type(object))
        return str(object)
    else:
        return str(object)


def deserialize_for_postgres(object):
    if isinstance(object, dict):
        type_cls = safe_get_type(object["type"])
        data = object["data"]
        # If data is a dict/object, convert it back to JSON string for from_json method
        if isinstance(data, (dict, list)):
            # Convert string enum values back to integers for from_json
            data = convert_enum_strings_to_values(data)
            data = json.dumps(data)
        return type_cls.from_json(data)
    else:
        try:
            return safe_eval_literal(object)
        except:
            return str(object)


def convert_enum_strings_to_values(data):
    """Convert human-readable enum strings back to integer values for deserialization."""
    if not isinstance(data, dict):
        return data

    # Create a copy to avoid modifying the original
    result = data.copy()

    # Define reverse enum mappings
    buffer_type_reverse_map = {"DRAM": 0, "L1": 1, "SYSTEM_MEMORY": 2, "L1_SMALL": 3, "TRACE": 4}

    memory_layout_reverse_map = {"INTERLEAVED": 0, "HEIGHT_SHARDED": 2, "WIDTH_SHARDED": 3, "BLOCK_SHARDED": 4}

    # Convert buffer_type back to integer if it's a string
    if "buffer_type" in result and isinstance(result["buffer_type"], str):
        result["buffer_type"] = buffer_type_reverse_map.get(result["buffer_type"], 0)

    # Convert memory_layout back to integer if it's a string
    if "memory_layout" in result and isinstance(result["memory_layout"], str):
        result["memory_layout"] = memory_layout_reverse_map.get(result["memory_layout"], 0)

    # Recursively process nested objects
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = convert_enum_strings_to_values(value)
        elif isinstance(value, list):
            result[key] = [convert_enum_strings_to_values(item) if isinstance(item, dict) else item for item in value]

    return result


def deserialize_vector_for_postgres(test_vector):
    """
    Deserialize a test vector that was serialized for PostgreSQL storage.
    """
    param_names = test_vector.keys()
    test_vector = [deserialize_for_postgres(test_vector[elem]) for elem in test_vector]
    test_vector = dict(zip(param_names, test_vector))
    return test_vector


def deserialize_vector(test_vector):
    param_names = test_vector.keys()
    test_vector = [deserialize(test_vector[elem]) for elem in test_vector]
    test_vector = dict(zip(param_names, test_vector))
    return test_vector
