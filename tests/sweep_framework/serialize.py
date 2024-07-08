# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ttnn import *
import tt_lib


def serialize(object):
    if "attribute_names" in dir(object) and "attribute_values" in dir(object):
        serialized_object = dict()
        serialized_object["type"] = str(type(object)).split("'")[1]
        attr_names = object.attribute_names
        attr_values = object.attribute_values()
        for i in range(len(attr_names)):
            if attr_names[i] == "shard_spec":
                if attr_values[i] == None:
                    continue
            serialized_object[attr_names[i]] = serialize(attr_values[i])
        return serialized_object
    elif type(object) == ttnn.CoreRange:
        serialized_object = dict()
        serialized_object["type"] = "tt_lib.tensor.CoreRange"
        serialized_object["start"] = f"ttnn.CoreCoord({object.start.x}, {object.start.y})"
        serialized_object["end"] = f"ttnn.CoreCoord({object.end.x}, {object.end.y})"
        return serialized_object
    elif type(object) == ttnn.CoreRangeSet:
        serialized_object = dict()
        serialized_object["type"] = "tt_lib.tensor.CoreRangeSet"
        core_ranges = [serialize(core_set) for core_set in object.core_ranges()]
        serialized_object["core_ranges"] = core_ranges
        return serialized_object
    else:
        return str(object)


def deserialize(object):
    if isinstance(object, dict):
        type = eval(object["type"])
        object.pop("type")
        if type == ttnn.CoreRangeSet:
            core_ranges = set()
            for core_range in object["core_ranges"]:
                core_ranges.add(deserialize(core_range))
            return ttnn.CoreRangeSet(core_ranges)
        deserialized_parameters = dict()
        for elem in object:
            deserialized_parameters[elem] = deserialize(object[elem])
        return type(**deserialized_parameters)
    else:
        try:
            return eval(object)
        except:
            return str(object)


def deserialize_vector(test_vector):
    param_names = test_vector.keys()
    test_vector = [deserialize(test_vector[elem]) for elem in test_vector]
    test_vector = dict(zip(param_names, test_vector))
    return test_vector
