# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import types
from functools import wraps


def decorator(f, name):
    @wraps(f)
    # Just wrapping C++ calls with a python side wrapper is enough
    # This will let them be picked up by the python settrace
    def tt_lib_wrapper(*args, **kwargs):
        local_name = name
        return f(*args, **kwargs)

    return tt_lib_wrapper


def callable_decorator(parrentObj):
    # This function is called on the C++ side which does not dump exceptions
    # Try catch is added to dump the exception to stdout
    try:
        for name in dir(parrentObj):
            obj = getattr(parrentObj, name)
            # TODO: Improve finding objects , __ search is a very bad idea
            if callable(obj) and "__" not in name:
                setattr(parrentObj, name, decorator(obj, name))
                callable_decorator(obj)
    except Exception as e:
        print(e)
        raise e
