"""
TTNN tracy hook (no-op).

TTNN expects this symbol when compiled with TRACY_ENABLE:
  tracy.ttnn_profiler_wrapper.callable_decorator

The extension calls:
  callable_decorator(m_device)

We provide a no-op that returns without modifying the device object.
"""

from __future__ import annotations


def callable_decorator(_device) -> None:
    return
