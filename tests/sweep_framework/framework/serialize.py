# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import json
from tests.sweep_framework.framework.statuses import VectorValidity, VectorStatus
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger
from ttnn._ttnn.tensor import DataType, Layout  # make eval("DataType.*"/"Layout.*") resolvable


def convert_enum_values_to_strings(data):
    """Convert enum integer values to human-readable strings"""
    if not isinstance(data, dict):
        return data

    # Create a copy to avoid modifying the original
    result = data.copy()

    # Get enum mappings dynamically from the actual enum classes
    buffer_type_map = {}
    try:
        BufferType = ttnn._ttnn.tensor.BufferType
        buffer_type_map = {member.value: name for name, member in BufferType.__members__.items()}
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load BufferType enum: {e}")

    memory_layout_map = {}
    try:
        TensorMemoryLayout = ttnn._ttnn.tensor.TensorMemoryLayout
        memory_layout_map = {member.value: name for name, member in TensorMemoryLayout.__members__.items()}
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load TensorMemoryLayout enum: {e}")

    # Convert buffer_type if present and mapping is available
    if "buffer_type" in result and isinstance(result["buffer_type"], int) and buffer_type_map:
        result["buffer_type"] = buffer_type_map.get(
            result["buffer_type"], f"UNKNOWN_BUFFER_TYPE_{result['buffer_type']}"
        )

    # Convert memory_layout if present and mapping is available
    if "memory_layout" in result and isinstance(result["memory_layout"], int) and memory_layout_map:
        result["memory_layout"] = memory_layout_map.get(
            result["memory_layout"], f"UNKNOWN_MEMORY_LAYOUT_{result['memory_layout']}"
        )

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
        type = eval(object["type"])
        return type.from_json(object["data"])
    else:
        try:
            return eval(object)
        except:
            return str(object)


def serialize_structured(object, warnings=[]):
    if "to_json" in dir(object):
        json_str = object.to_json()
        try:
            # Parse the JSON string
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


def deserialize_structured(object):
    if isinstance(object, dict):
        type = eval(object["type"])
        data = object["data"]
        # If data is a dict/object, convert it back to JSON string for from_json method
        if isinstance(data, (dict, list)):
            # Convert string enum values back to integers for from_json
            data = convert_enum_strings_to_values(data)
            data = json.dumps(data)
        return type.from_json(data)
    else:
        try:
            return eval(object)
        except:
            return str(object)


def convert_enum_strings_to_values(data):
    """Convert human-readable enum strings back to integer values for deserialization."""
    if not isinstance(data, dict):
        return data

    # Create a copy to avoid modifying the original
    result = data.copy()

    # Get reverse enum mappings dynamically from the actual enum classes
    buffer_type_reverse_map = {}
    try:
        BufferType = ttnn._ttnn.tensor.BufferType
        buffer_type_reverse_map = {name: member.value for name, member in BufferType.__members__.items()}
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load BufferType enum for reverse mapping: {e}")

    memory_layout_reverse_map = {}
    try:
        TensorMemoryLayout = ttnn._ttnn.tensor.TensorMemoryLayout
        memory_layout_reverse_map = {name: member.value for name, member in TensorMemoryLayout.__members__.items()}
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load TensorMemoryLayout enum for reverse mapping: {e}")

    # Convert buffer_type back to integer if it's a string and mapping is available
    if "buffer_type" in result and isinstance(result["buffer_type"], str) and buffer_type_reverse_map:
        if result["buffer_type"] in buffer_type_reverse_map:
            result["buffer_type"] = buffer_type_reverse_map[result["buffer_type"]]
        else:
            logger.warning(f"Unknown buffer_type string: {result['buffer_type']}")

    # Convert memory_layout back to integer if it's a string and mapping is available
    if "memory_layout" in result and isinstance(result["memory_layout"], str) and memory_layout_reverse_map:
        if result["memory_layout"] in memory_layout_reverse_map:
            result["memory_layout"] = memory_layout_reverse_map[result["memory_layout"]]
        else:
            logger.warning(f"Unknown memory_layout string: {result['memory_layout']}")

    # Recursively process nested objects
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = convert_enum_strings_to_values(value)
        elif isinstance(value, list):
            result[key] = [convert_enum_strings_to_values(item) if isinstance(item, dict) else item for item in value]

    return result


def deserialize_vector_structured(test_vector):
    """
    Deserialize a test vector from a human-readable JSON to TTNN enums
    """
    param_names = test_vector.keys()
    test_vector = [deserialize_structured(test_vector[elem]) for elem in test_vector]
    test_vector = dict(zip(param_names, test_vector))
    return test_vector
