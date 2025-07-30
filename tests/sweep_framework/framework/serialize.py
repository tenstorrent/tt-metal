# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ttnn import *
import json
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


def convert_enum_values_to_strings(data):
    """Convert enum integer values to human-readable strings for PostgreSQL storage."""
    if not isinstance(data, dict):
        return data

    # Create a copy to avoid modifying the original
    result = data.copy()

    # Define enum mappings - from tt_metal/api/tt-metalium/buffer_types.hpp
    buffer_type_map = {0: "DRAM", 1: "L1", 2: "SYSTEM_MEMORY", 3: "L1_SMALL", 4: "TRACE"}

    memory_layout_map = {0: "INTERLEAVED", 2: "HEIGHT_SHARDED", 3: "WIDTH_SHARDED", 4: "BLOCK_SHARDED"}

    shard_orientation_map = {0: "ROW_MAJOR", 1: "COL_MAJOR"}

    shard_distribution_strategy_map = {0: "ROUND_ROBIN_1D", 1: "GRID_2D"}

    shard_mode_map = {0: "PHYSICAL", 1: "LOGICAL"}

    # Define enum mappings - from ttnn/api/ttnn/tensor/types.hpp
    data_type_map = {
        0: "BFLOAT16",
        1: "FLOAT32",
        2: "UINT32",
        3: "BFLOAT8_B",
        4: "BFLOAT4_B",
        5: "UINT8",
        6: "UINT16",
        7: "INT32",
        8: "INVALID",
    }

    # Define enum mappings - from ttnn/api/ttnn/tensor/enum_types.hpp
    layout_map = {0: "ROW_MAJOR", 1: "TILE", 2: "INVALID"}

    # Define enum mappings - from ttnn/api/ttnn/tensor/types.hpp
    storage_type_map = {0: "HOST", 1: "DEVICE"}

    # Define enum mappings - from tt_metal/impl/flatbuffer/base_types.fbs
    math_fidelity_map = {0: "LoFi", 2: "HiFi2", 3: "HiFi3", 4: "HiFi4", 255: "Invalid"}

    # Convert buffer_type if present
    if "buffer_type" in result and isinstance(result["buffer_type"], int):
        result["buffer_type"] = buffer_type_map.get(result["buffer_type"], f"UNKNOWN_{result['buffer_type']}")

    # Convert memory_layout if present
    if "memory_layout" in result and isinstance(result["memory_layout"], int):
        result["memory_layout"] = memory_layout_map.get(result["memory_layout"], f"UNKNOWN_{result['memory_layout']}")

    # Convert shard_orientation if present
    if "orientation" in result and isinstance(result["orientation"], int):
        result["orientation"] = shard_orientation_map.get(result["orientation"], f"UNKNOWN_{result['orientation']}")

    # Convert shard_distribution_strategy if present
    if "shard_distribution_strategy" in result and isinstance(result["shard_distribution_strategy"], int):
        result["shard_distribution_strategy"] = shard_distribution_strategy_map.get(
            result["shard_distribution_strategy"], f"UNKNOWN_{result['shard_distribution_strategy']}"
        )

    # Convert shard_mode if present
    if "shard_mode" in result and isinstance(result["shard_mode"], int):
        result["shard_mode"] = shard_mode_map.get(result["shard_mode"], f"UNKNOWN_{result['shard_mode']}")

    # Convert data_type if present
    if "data_type" in result and isinstance(result["data_type"], int):
        result["data_type"] = data_type_map.get(result["data_type"], f"UNKNOWN_{result['data_type']}")

    # Convert layout if present
    if "layout" in result and isinstance(result["layout"], int):
        result["layout"] = layout_map.get(result["layout"], f"UNKNOWN_{result['layout']}")

    # Convert storage_type if present
    if "storage_type" in result and isinstance(result["storage_type"], int):
        result["storage_type"] = storage_type_map.get(result["storage_type"], f"UNKNOWN_{result['storage_type']}")

    # Convert math_fidelity if present
    if "math_fidelity" in result and isinstance(result["math_fidelity"], int):
        result["math_fidelity"] = math_fidelity_map.get(result["math_fidelity"], f"UNKNOWN_{result['math_fidelity']}")

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

    # Define reverse enum mappings - from tt_metal/api/tt-metalium/buffer_types.hpp
    buffer_type_reverse_map = {"DRAM": 0, "L1": 1, "SYSTEM_MEMORY": 2, "L1_SMALL": 3, "TRACE": 4}

    memory_layout_reverse_map = {"INTERLEAVED": 0, "HEIGHT_SHARDED": 2, "WIDTH_SHARDED": 3, "BLOCK_SHARDED": 4}

    shard_orientation_reverse_map = {"ROW_MAJOR": 0, "COL_MAJOR": 1}

    shard_distribution_strategy_reverse_map = {"ROUND_ROBIN_1D": 0, "GRID_2D": 1}

    shard_mode_reverse_map = {"PHYSICAL": 0, "LOGICAL": 1}

    # Define reverse enum mappings - from ttnn/api/ttnn/tensor/types.hpp
    data_type_reverse_map = {
        "BFLOAT16": 0,
        "FLOAT32": 1,
        "UINT32": 2,
        "BFLOAT8_B": 3,
        "BFLOAT4_B": 4,
        "UINT8": 5,
        "UINT16": 6,
        "INT32": 7,
        "INVALID": 8,
    }

    # Define reverse enum mappings - from ttnn/api/ttnn/tensor/enum_types.hpp
    layout_reverse_map = {"ROW_MAJOR": 0, "TILE": 1, "INVALID": 2}

    # Define reverse enum mappings - from ttnn/api/ttnn/tensor/types.hpp
    storage_type_reverse_map = {"HOST": 0, "DEVICE": 1}

    # Define reverse enum mappings - from tt_metal/impl/flatbuffer/base_types.fbs
    math_fidelity_reverse_map = {"LoFi": 0, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4, "Invalid": 255}

    # Convert buffer_type back to integer if it's a string
    if "buffer_type" in result and isinstance(result["buffer_type"], str):
        result["buffer_type"] = buffer_type_reverse_map.get(result["buffer_type"], 0)

    # Convert memory_layout back to integer if it's a string
    if "memory_layout" in result and isinstance(result["memory_layout"], str):
        result["memory_layout"] = memory_layout_reverse_map.get(result["memory_layout"], 0)

    # Convert shard_orientation back to integer if it's a string
    if "orientation" in result and isinstance(result["orientation"], str):
        result["orientation"] = shard_orientation_reverse_map.get(result["orientation"], 0)

    # Convert shard_distribution_strategy back to integer if it's a string
    if "shard_distribution_strategy" in result and isinstance(result["shard_distribution_strategy"], str):
        result["shard_distribution_strategy"] = shard_distribution_strategy_reverse_map.get(
            result["shard_distribution_strategy"], 0
        )

    # Convert shard_mode back to integer if it's a string
    if "shard_mode" in result and isinstance(result["shard_mode"], str):
        result["shard_mode"] = shard_mode_reverse_map.get(result["shard_mode"], 0)

    # Convert data_type back to integer if it's a string
    if "data_type" in result and isinstance(result["data_type"], str):
        result["data_type"] = data_type_reverse_map.get(result["data_type"], 0)

    # Convert layout back to integer if it's a string
    if "layout" in result and isinstance(result["layout"], str):
        result["layout"] = layout_reverse_map.get(result["layout"], 0)

    # Convert storage_type back to integer if it's a string
    if "storage_type" in result and isinstance(result["storage_type"], str):
        result["storage_type"] = storage_type_reverse_map.get(result["storage_type"], 0)

    # Convert math_fidelity back to integer if it's a string
    if "math_fidelity" in result and isinstance(result["math_fidelity"], str):
        result["math_fidelity"] = math_fidelity_reverse_map.get(result["math_fidelity"], 0)

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
