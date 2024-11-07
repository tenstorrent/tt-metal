# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import gc


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


def traced(callable):
    """
    Test it locally, get it into ttnn mainline
    """
    # TODO: release trace on delete or ???
    trace_id = None
    args_device = None
    kwargs_device = None
    outputs = None

    def create_device_inputs(*args, **kwargs):
        # allocate device tensors for each arg which is on host
        # don't copy
        nonlocal args_device
        nonlocal kwargs_device

    def copy_inputs_to_device(*args, **kwargs):
        # copy any host tensors to device

        # Check that kwargs keys matches kwargs_device keys
        # check that args len matches args_device len
        nonlocal args_device
        nonlocal kwargs_device
        pass

    def wrapper(self, *args, **kwargs):
        nonlocal trace_id
        nonlocal outputs
        if not trace_id:
            create_device_inputs(args, kwargs)
            copy_inputs_to_device(args, kwargs)
            ret = callable(self, *args, **kwargs)
            outputs = ret
            trace_id = ttnn.capture_trace(...)
            callable(self, *args, **kwargs)
            ttnn.end_trace(...)
            return ret
        # check that inputs, outputs are host tensors
        # or if an input is on device, do nothing
        # copy new inputs to inputs, return outputs
        copy_inputs_to_device(args, kwargs)
        ttnn.execute_trace(trace_id)
        return outputs

    return wrapper
