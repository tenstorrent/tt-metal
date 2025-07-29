# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ttnn import *
import json
from tests.sweep_framework.framework.statuses import VectorValidity
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


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
            data = json.dumps(data)
        return type.from_json(data)
    else:
        try:
            return eval(object)
        except:
            return str(object)


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
