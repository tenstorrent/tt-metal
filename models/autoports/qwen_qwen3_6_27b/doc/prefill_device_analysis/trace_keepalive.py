# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Experimental ownership control for a small trace embedded in eager work.

Keep the yielded list alive until the associated trace has been released.
This retains Python-visible operation outputs; temporary buffers created and
freed entirely inside C++ composite operations remain outside its coverage.
"""

from contextlib import contextmanager

import ttnn


@contextmanager
def preserve_trace_tensors():
    """Hold operation results and suppress Python deallocation during capture.

    Usage::

        with preserve_trace_tensors() as keepalive:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs = recurrence(*buffers, **kwargs)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_cache[key] = (buffers, outputs, trace, keepalive)

    Warm the exact recurrence and input-copy variants before entering. This
    temporarily changes process-wide TTNN behavior, so use only in a serialized
    experiment. It does not change allocation-tracker settings or acknowledgments.
    """
    keepalive = []
    seen = set()

    def retain(value):
        if isinstance(value, ttnn.Tensor):
            identity = id(value)
            if identity not in seen:
                seen.add(identity)
                keepalive.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                retain(item)
        elif isinstance(value, dict):
            for item in value.values():
                retain(item)

    def after_operation(operation, args, kwargs, output):
        retain(output)

    def defer_deallocation(*args, **kwargs):
        # Also retain arguments in case a Python helper bypassed an op hook.
        retain(args)
        retain(kwargs)

    original_deallocate = ttnn.deallocate
    original_fast_runtime = ttnn.CONFIG.enable_fast_runtime_mode
    try:
        # FastOperation bypasses hooks unless it routes through Operation.
        ttnn.CONFIG.enable_fast_runtime_mode = False
        ttnn.deallocate = defer_deallocation
        with ttnn.register_post_operation_hook(after_operation):
            yield keepalive
    finally:
        ttnn.deallocate = original_deallocate
        ttnn.CONFIG.enable_fast_runtime_mode = original_fast_runtime
