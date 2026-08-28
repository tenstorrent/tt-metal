# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The mesh every entrypoint OUTSIDE pytest runs this pipeline on.

`tt/pipeline.py` never opens a device: it runs on the `device` handed to
`build_pipeline`, and inside pytest that device comes from the `mesh_device`
fixture, which is the sole opener. But the two demos, the host-op observer and the
trace probe all start from a bare process with no fixture, so SOMETHING has to own
a mesh for the duration of the call. That owner is here -- one file, outside
`tt/`, so the pipeline package itself stays free of device ownership and there is
exactly one place to look when these mesh parameters need to match the fixture's.

The parameters below mirror `tests/e2e/test_trace_contract.py::_DEVICE_PARAMS`
exactly, so the standalone runs exercise the same device the gate tests do.
"""
from __future__ import annotations

import contextlib
import os

import ttnn

# The mesh the components graduated at: 1 x 8, TP=8.
SELFTEST_TP = int(os.environ.get("TT_HW_PLANNER_SHARD_TP", "8"))

# Depth for the standalone self-tests. Host-op residency and trace capture are
# properties of the ops a layer runs, not of how many times the stack repeats it,
# so a capped stack exercises the same op set in a fraction of the time. The FULL
# depth is what tests/e2e/test_e2e_pipeline.py gates correctness on.
SELFTEST_LAYERS = int(os.environ.get("TT_SELFTEST_LAYERS", "4"))

L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = 90000000
FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D


@contextlib.contextmanager
def own_a_mesh(tp: int = SELFTEST_TP):
    """A mesh device for the length of one standalone self-test, then closed.

    The order is the `mesh_device` fixture's own: fabric is configured BEFORE the
    mesh is opened (the row-parallel all_reduces and the vocab all_gather need it)
    and reset to DISABLED after it closes. `set_fabric_config`'s defaults are the
    fixture's values -- STRICT_INIT reliability, tensix and UDM disabled -- so this
    is the same fabric the gate tests run on.
    """
    ttnn.set_fabric_config(FABRIC_CONFIG)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, int(tp)),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        dispatch_core_config=ttnn.DispatchCoreConfig(None, None, None),
    )
    try:
        yield device
    finally:
        for submesh in device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
