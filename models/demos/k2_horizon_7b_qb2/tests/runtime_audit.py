"""Fail a forward call if it reaches a host tensor/conversion implementation."""

import functools
import sys
from contextlib import contextmanager

import ttnn


@contextmanager
def device_only():
    names = ("from_torch", "to_torch", "as_tensor", "to_device", "copy_host_to_device_tensor")
    saved = {name: getattr(ttnn, name) for name in names}
    previous = sys.getprofile()

    def forbidden(*args, **kwargs):
        raise AssertionError("Host tensor boundary inside a decoder forward")

    def profile(frame, event, arg):
        if event == "call":
            path = frame.f_code.co_filename
            if "/torch/" in path or "/numpy/" in path:
                raise AssertionError(f"Host tensor execution inside forward: {path}:{frame.f_code.co_name}")
        if event == "c_call":
            module = getattr(arg, "__module__", "") or ""
            if module.startswith(("torch", "numpy")):
                raise AssertionError(f"Host tensor execution inside forward: {module}")

    try:
        for name in names:
            setattr(ttnn, name, forbidden)
        sys.setprofile(profile)
        yield
    finally:
        sys.setprofile(previous)
        for name, value in saved.items():
            setattr(ttnn, name, value)


def instrument(layer):
    counts = {"prefill_forward": 0, "decode_forward": 0}
    for name in counts:
        method = getattr(layer, name)

        def decorate(method, name):
            @functools.wraps(method)
            def forward(*args, **kwargs):
                with device_only():
                    result = method(*args, **kwargs)
                counts[name] += 1
                return result

            return forward

        setattr(layer, name, decorate(method, name))
    return counts
