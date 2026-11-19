# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prove the current four-chip runtime rejects a live post-capture hazard."""

import gc
import os

assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10_000_000)
trace = None
try:
    source = ttnn.from_torch(
        torch.ones(1, 1, 32, 32),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    warm = ttnn.add(source, source)
    del warm
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    output = ttnn.add(source, source)
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    # The single trace owns this output; no intervening graph consumes it.
    trace_allocation_tracker.acknowledge_corruptible(output)
    hazard = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    except RuntimeError as error:
        assert "allocat" in str(error).lower(), str(error)
        print("TRACKER_POSITIVE_CONTROL_REJECTED", str(error).splitlines()[0], flush=True)
    else:
        raise AssertionError("Live hazard was not rejected")
    del hazard
    gc.collect()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    for shard in ttnn.get_device_tensors(output):
        assert torch.equal(ttnn.to_torch(shard), torch.full((1, 1, 32, 32), 2.0, dtype=torch.bfloat16))
    print("TRACKER_CLEAN_REPLAY_PASS", flush=True)
finally:
    if trace is not None:
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)
