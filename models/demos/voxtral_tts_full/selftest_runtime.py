# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The device opener for the STANDALONE selftest entry points in `tt/pipeline.py`.

`scripts/tt_hw_planner/_host_op_probe.py` and `_trace_capture_probe.py` import
`<demo>.tt.pipeline` in a fresh process and call `mod.host_op_selftest()` /
`mod.trace_capture_selftest()` with NO arguments.  There is no pytest fixture in that process to
hand them a device, so those entry points have to obtain one themselves.

They cannot obtain it inside `tt/`.  The device-ownership rule is that the pipeline runs on the
device passed into `build_pipeline`, and a second ad-hoc open anywhere in the importable pipeline
package creates a competing device with a different command-queue count -- the
`id < mesh_command_queues_.size()` fatal that breaks trace.  So the ONE open the standalone
observers need lives here, outside `tt/`, and is reached only when a caller passed no device.
Every other caller (the pytest fixtures, `demo/demo_tts.py`) still opens its own device and passes
it in, exactly as before.
"""

from __future__ import annotations

import contextlib

import ttnn


@contextlib.contextmanager
def selftest_device(device_id=0, trace_region_size=None):
    """One device, opened with a trace region so a capture has somewhere to live, and closed again.

    `trace_region_size=None` takes the pipeline's own default, which is sized for its largest
    stage -- the same value `tests/e2e/conftest.py` opens with, so the standalone observers and the
    test session exercise identical device configuration."""
    from models.demos.voxtral_tts_full.tt import pipeline as P

    dev = ttnn.open_device(
        device_id=device_id,
        trace_region_size=P.DEFAULT_TRACE_REGION_SIZE if trace_region_size is None else trace_region_size,
    )
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


def with_pipeline(fn, device_id=0, trace_region_size=None, **build_kwargs):
    """Open a device, build the pipeline on it, hand `(pipe, device)` to `fn`, close, return.

    `build_kwargs` go straight to `build_pipeline`, so a caller that only needs the op set (a trace
    capture) can cap the repeated stacks while a caller that needs the whole chain does not."""
    from models.demos.voxtral_tts_full.tt import pipeline as P

    with selftest_device(device_id, trace_region_size) as dev:
        return fn(P.build_pipeline(dev, **build_kwargs), dev)
