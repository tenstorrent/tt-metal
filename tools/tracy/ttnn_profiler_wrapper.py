# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import types
from functools import wraps


def decorator(f, name):
    @wraps(f)
    def tt_lib_wrapper(*args, **kwargs):
        local_name = name
        import ttnn

        ttnn.start_tracy_zone("ttnn_profiler_wrapper.py", f"python::{name}", 0)
        try:
            return f(*args, **kwargs)
        finally:
            ttnn.stop_tracy_zone(f"python::{name}")

    # Allow inspection code such as in ttnn/ttnn/ttl/tensor/__init__.py to
    # detect that this is a wrapped C function and treat it appropriately
    tt_lib_wrapper.profiler_wrapped_function = f
    return tt_lib_wrapper


def callable_decorator(parentObj):
    # This function is called on the c++ side which does not dump exceptions
    # Try catch is added to dump the exception to stdout
    try:
        for name in dir(parentObj):
            obj = getattr(parentObj, name)
            # TODO: Improve finding objects , __ search is a very bad idea
            if not isinstance(obj, type) and callable(obj) and "__" not in name:
                setattr(parentObj, name, decorator(obj, name))
                callable_decorator(obj)
    except Exception as e:
        print(e)
        raise e
