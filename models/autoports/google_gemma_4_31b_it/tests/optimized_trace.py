"""Balanced trace lifecycle for candidate validation, including body failures."""

import ttnn


def capture(mesh, forward):
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        result = forward()
    except BaseException:
        # Body-time validation can fail with dispatch bypass still enabled.
        # End and discard the partial trace; it must never be replayed.
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    return trace, result
