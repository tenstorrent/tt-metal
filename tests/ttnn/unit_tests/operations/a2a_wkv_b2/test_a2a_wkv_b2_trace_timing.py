# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-call wall time of a2a_wkv_b2 under trace replay (all 8 chips run concurrently).

Captures one call in a trace, replays it N times back-to-back and reports mean wall-clock us per
replay. Dispatch is off the critical path in a trace, so this is close to the per-call device time
of the slowest chip (plus cross-chip launch skew). Used as the verifier's perf baseline.
"""

import time

import pytest
import ttnn

from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs

N_REPLAYS = 50


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_a2a_wkv_b2_trace_timing(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2(attn, w))  # compile + semaphore creation (eager)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(N_REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REPLAYS * 1e6
    ttnn.release_trace(mesh_device, tid)
    print(f"\nTRACE_TIMING a2a_wkv_b2: {us:.1f} us per call (mean of {N_REPLAYS} replays)")
