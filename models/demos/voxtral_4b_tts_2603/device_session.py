# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device ownership for the STANDALONE selftest entrypoints of this demo.

This module sits OUTSIDE `tt/` on purpose.

`tt/` is the pipeline package, and the pipeline must never open a device: it runs on the one
handed to `build_pipeline(device, ...)`, and the test fixture is the sole opener. A second,
ad-hoc open inside the importable pipeline path creates a competing device with a different
command-queue count -- the `id < mesh_command_queues_.size()` fatal that breaks trace.

But the bring-up tool's observers (`scripts/tt_hw_planner/_host_op_probe.py` and
`_trace_capture_probe.py`) import `tt.pipeline` in a FRESH process with no device open and call
`host_op_selftest()` / `trace_capture_selftest()` with zero arguments. Something has to own a
device for those. Keeping that ownership here draws the line exactly where the rule does: the
pipeline never opens a device, the standalone entry does -- the same carve-out a
`if __name__ == "__main__"` self-test gets.
"""
from __future__ import annotations

import contextlib
import os

import ttnn

# Matches tests/e2e/test_trace_and_host_ops.py, which is the fixture the trace contract was
# validated under; a selftest that opened the device differently would not be testing the same
# thing the suite does.
L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = 200 * 1024 * 1024


@contextlib.contextmanager
def selftest_device(device_id: int = int(os.environ.get("VOXTRAL_DEVICE_ID", "0")), trace_region_size: int = TRACE_REGION_SIZE):
    """Open ONE single-command-queue device for a standalone selftest, and always close it."""
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=trace_region_size,
        num_command_queues=1,
    )
    try:
        yield device
    finally:
        ttnn.close_device(device)
