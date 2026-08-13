# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device owner for the ZERO-ARG selftest hooks of this demo.

`tt/pipeline.py` never opens a device: `build_pipeline(device, ...)` runs on the
device it is handed, and the pytest fixture is the sole opener for every test.
But two tool-facing hooks are invoked with NO arguments from a bare
subprocess — `scripts/tt_hw_planner/_trace_capture_probe.py` calls
`trace_capture_selftest()` and `scripts/tt_hw_planner/_host_op_probe.py` calls
`host_op_selftest()` — so in that context something has to own a device.

That opener lives HERE, one level outside the `tt/` package, so the pipeline
package itself stays free of `ttnn.open_device`: exactly one device is open at a
time, with one command queue and a trace region, which is what keeps trace
capture from tripping over a competing device. Both hooks also accept an
explicit `device=`, and the tests pass the fixture device so this module is not
used from pytest at all.
"""
from __future__ import annotations

import os

L1_SMALL_SIZE = 24576
# The prefill stage traces 26 decoder layers at the full [1, C, 3072] shape; the
# region has to hold every command in that chain. Sized with headroom (DRAM is
# far larger than the weights) and overridable for a smaller board.
TRACE_REGION_SIZE = int(os.environ.get("TT_E2E_TRACE_REGION", 600_000_000))


def open_selftest_device(trace_region_size: int = None):
    """Open THE single device the zero-arg selftests run on."""
    import ttnn

    return ttnn.open_device(
        device_id=int(os.environ.get("TT_E2E_DEVICE_ID", 0)),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE if trace_region_size is None else int(trace_region_size),
    )


def close_selftest_device(device) -> None:
    import ttnn

    try:
        ttnn.close_device(device)
    except Exception as exc:  # noqa: BLE001 - teardown must not mask the verdict
        print(f"[voxtral-e2e] close_device failed: {type(exc).__name__}: {exc}", flush=True)
