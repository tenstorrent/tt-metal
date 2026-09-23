# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device ownership for this demo — deliberately OUTSIDE `tt/`.

`tt/pipeline.py` never opens a device: it runs on the `device` handed to
`build_pipeline`, so the caller (the pytest fixture, the demo entrypoint) is the
sole opener and there is never a second, competing device with a different
command-queue count. This module is that caller's helper — one place that knows
how the 1x4 Blackhole mesh is opened, so the demo and the zero-arg
`host_op_selftest()` probe entry agree on l1_small_size / trace_region_size /
fabric config instead of each rolling their own.
"""
from __future__ import annotations

import ttnn

# The trace region the per-stage trace contract needs; kept here so a caller that
# opens the mesh for a trace run cannot forget it.
DEFAULT_TRACE_REGION_SIZE = 90 * 1024 * 1024
DEFAULT_L1_SMALL_SIZE = 24576


def open_mesh(
    rows: int = 1,
    cols: int = 4,
    l1_small_size: int = DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE,
):
    """Open the TP=4 mesh — fabric first, because the graduated attention/MLP stubs
    all_reduce on cluster axis 1 — falling back to whatever is actually available
    and saying so."""
    want = rows * cols
    have = ttnn.get_num_devices()
    if have < want:
        print(f"[mesh] only {have} device(s) available, wanted {want} — falling back", flush=True)
        rows, cols = 1, have
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    kwargs = {"l1_small_size": l1_small_size}
    if trace_region_size:
        kwargs["trace_region_size"] = trace_region_size
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **kwargs)
    print(f"[mesh] opened MeshShape({rows}, {cols})  DP={rows} TP={cols}", flush=True)
    return device


def close_mesh(device) -> None:
    ttnn.close_mesh_device(device)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
