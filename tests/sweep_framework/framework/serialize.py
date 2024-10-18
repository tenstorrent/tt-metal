# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ttnn import *
import tt_lib
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


def deserialize_vector(test_vector):
    param_names = test_vector.keys()
    test_vector = [deserialize(test_vector[elem]) for elem in test_vector]
    test_vector = dict(zip(param_names, test_vector))
    return test_vector
