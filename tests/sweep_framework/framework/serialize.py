# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ast
import json
from typing import Any

import ttnn

from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger

TTNN_NAME = ttnn.__name__


# =============================================================================
# TYPE REGISTRY: Safe type lookup for deserialization (replaces eval)
# =============================================================================

_TTNN_TYPE_REGISTRY: dict[str, type] = {}


def _build_type_registry() -> None:
    """Build whitelist of known ttnn types for safe deserialization."""
    serializable_types = ["MemoryConfig", "ShardSpec", "CoreRangeSet", "CoreRange", "CoreCoord"]
    for type_name in serializable_types:
        if hasattr(ttnn, type_name):
            t = getattr(ttnn, type_name)
            _TTNN_TYPE_REGISTRY[type_name] = t
            _TTNN_TYPE_REGISTRY[f"ttnn.{type_name}"] = t
        try:
            t = getattr(ttnn._ttnn.tensor, type_name)
            _TTNN_TYPE_REGISTRY[f"ttnn._ttnn.tensor.{type_name}"] = t
        except AttributeError:
            # Some ttnn builds may not expose these optional types on ttnn._ttnn.tensor; skip them.
            logger.debug(f"Optional ttnn._ttnn.tensor type '{type_name}' not available")


def _resolve_type(type_name: str) -> type:
    """Resolve type name to type using whitelist registry."""
    if type_name in _TTNN_TYPE_REGISTRY:
        return _TTNN_TYPE_REGISTRY[type_name]
    raise ValueError(f"Unknown type '{type_name}'. Add to _TTNN_TYPE_REGISTRY if valid.")


_build_type_registry()


def _is_serialized_ttnn_object(obj: dict) -> bool:
    """Check if a dict represents a serialized ttnn object.

    A valid serialized ttnn object must have:
    - Exactly two keys: "type" and "data"
    - "type" must be a string
    - "type" must be a known type in the registry
    """
    return (
        isinstance(obj, dict)
        and obj.keys() == {"type", "data"}
        and isinstance(obj.get("type"), str)
        and obj["type"] in _TTNN_TYPE_REGISTRY
    )


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


def _ttnn_type_from_name(type_name):
    uq_type_name = type_name.split(".")[-1]
    the_type = None
    try:
        the_type = getattr(ttnn, uq_type_name)
    except AttributeError as e:
        logger.debug(f"Hopefully not an enum {e}")

    return the_type


def _deserialize_ttnn_enum(obj_name: str) -> Any | None:
    """Safely deserialize a ttnn enum member from its string representation.

    Example: "ttnn.DataType.BFLOAT16" -> ttnn.DataType.BFLOAT16

    Uses getattr() instead of eval() for safe attribute access.
    """
    parts = [p for p in obj_name.split(".") if p != TTNN_NAME]
    if len(parts) < 2:
        return None
    enum_type = _ttnn_type_from_name(parts[0])
    if enum_type is None:
        return None
    member_name = parts[1]
    if not hasattr(enum_type, member_name):
        return None
    return getattr(enum_type, member_name)


def _safe_literal_eval(s: str) -> Any:
    """Safely parse Python literals without code execution.

    Only parses: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None.
    Does NOT execute arbitrary code.

    Args:
        s: String representation of a Python literal

    Returns:
        The parsed Python object, or the original string if parsing fails.
    """
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s


def deserialize(obj: Any) -> Any:
    """Deserialize an object from its serialized representation.

    Handles:
    - Dicts with {"type": ..., "data": ...} -> ttnn object via from_json
    - Dotted strings -> ttnn enum members
    - Literal strings -> Python primitives via ast.literal_eval
    """
    try:
        if isinstance(obj, dict):
            if _is_serialized_ttnn_object(obj):
                obj_type = _resolve_type(obj["type"])
                return obj_type.from_json(obj["data"])
            else:
                raise ValueError(f"Dict does not match serialized ttnn object format: {obj}")

        if isinstance(obj, str):
            # Try enum deserialization for dotted names
            if "." in obj:
                maybe_enum = _deserialize_ttnn_enum(obj)
                if maybe_enum is not None:
                    return maybe_enum

            # Try literal parsing for primitives (safe, no code execution)
            return _safe_literal_eval(obj)

        return obj

    except Exception as e:
        logger.exception(f"deserialize failed {e}")
        raise


def serialize_structured(object, warnings=[]):
    if isinstance(object, (str, int, float, bool, type(None))):
        return object
    elif isinstance(object, dict):
        return {k: serialize_structured(v, warnings) for k, v in object.items()}
    elif isinstance(object, list):
        return [serialize_structured(item, warnings) for item in object]
    elif "to_json" in dir(object):
        json_str = object.to_json()
        try:
            parsed_data = json.loads(json_str)
            parsed_data = convert_enum_values_to_strings(parsed_data)
            return {"type": str(type(object)).split("'")[1], "data": parsed_data}
        except (json.JSONDecodeError, TypeError):
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
    try:
        # Handle JSON-native primitives directly (no deserialization needed)
        if isinstance(object, (int, float, bool, type(None))):
            return object

        # Handle lists by recursively deserializing elements
        if isinstance(object, list):
            return [deserialize_structured(item) for item in object]

        # Handle dicts - check if it's a serialized ttnn object or a plain dict
        if isinstance(object, dict):
            if _is_serialized_ttnn_object(object):
                # This is a serialized ttnn object - use type registry instead of eval
                obj_type = _resolve_type(object["type"])
                data = object["data"]
                # If data is a dict/object, convert it back to JSON string for from_json method
                if isinstance(data, (dict, list)):
                    # Convert string enum values back to integers for from_json
                    data = convert_enum_strings_to_values(data)
                    data = json.dumps(data)
                return obj_type.from_json(data)
            else:
                # Plain dict - recursively deserialize values
                return {k: deserialize_structured(v) for k, v in object.items()}

        # Handle strings
        if isinstance(object, str):
            if "." in object:
                maybe_enum = _deserialize_ttnn_enum(object)
                if maybe_enum is not None:
                    return maybe_enum
            elif object in ["sum", "mean", "max", "min", "std", "var"]:
                return object
            # Safe literal parsing (no code execution)
            return _safe_literal_eval(object)

        # Fallback - return as-is
        return object
    except Exception as e:
        logger.exception(f"Deserialize structured failed {e}")
        raise


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
