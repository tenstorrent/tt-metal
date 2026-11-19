# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prove the allocation tracker rejects a live post-capture device allocation."""

import os

import torch

import ttnn

assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
trace = None
try:
    x = ttnn.from_torch(torch.ones(32, 32), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output = ttnn.empty_like(x)
    ttnn.add(x, x, output_tensor=output)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    ttnn.add(x, x, output_tensor=output)
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
    live_unsafe = ttnn.empty_like(x)
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
    except RuntimeError as error:
        print("EXPECTED_TRACKER_REJECTION", str(error), flush=True)
    else:
        raise AssertionError("Live post-capture allocation escaped tracker")
finally:
    if trace is not None:
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)
