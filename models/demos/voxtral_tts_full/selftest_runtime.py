# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device ownership for the STANDALONE self-test probes.

`scripts/tt_hw_planner/_host_op_probe.py` and `_trace_capture_probe.py` import `tt/pipeline.py`
into a fresh process and call `host_op_selftest()` / `trace_capture_selftest()` with NO arguments.
There is no pytest fixture in that process, so those entry points have to acquire a device
themselves.

THE OPENER DELIBERATELY DOES NOT LIVE IN `tt/`. The pipeline package runs on whatever device is
handed to `build_pipeline`, and inside a test session the pytest fixture is the ONLY opener: it
opens once with the command-queue count and `trace_region_size` the trace lever needs. A second
opener reachable from the pipeline's own import graph is how a competing device with a different
command-queue count gets created -- the `id < mesh_command_queues_.size()` fatal that kills trace
capture. Keeping it in this module means nothing under `tt/` can open a device even by accident,
while the out-of-process probes still get one.
"""
from __future__ import annotations

import contextlib

import ttnn


@contextlib.contextmanager
def standalone_device(trace_region_size: int, device_id: int = 0):
    """Open a device for the duration of ONE standalone self-test, then close it.

    `trace_region_size` is the pipeline's own `TRACE_REGION_SIZE`, so a probe that goes on to
    capture a trace gets the same region the test fixture would have given it.
    """
    device = ttnn.open_device(device_id=int(device_id), trace_region_size=int(trace_region_size))
    try:
        yield device
    finally:
        ttnn.close_device(device)
