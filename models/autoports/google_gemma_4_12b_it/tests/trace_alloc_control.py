# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Positive control for TT_METAL_TRACE_ALLOC_TRACKING=1: an allocation kept alive after capture must
make execute_trace raise, proving the checker is active in this environment."""

import torch

import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
x = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
ttnn.add(x, x)
tid = ttnn.begin_trace_capture(mesh, cq_id=0)
y = ttnn.add(x, x)
ttnn.end_trace_capture(mesh, tid, cq_id=0)
late = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
try:
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    print("CONTROL: execute_trace accepted an unsafe live allocation (checker inactive)")
except RuntimeError as e:
    print("CONTROL: checker active, execute_trace refused:", str(e).splitlines()[0][:200])
ttnn.release_trace(mesh, tid)
ttnn.close_mesh_device(mesh)
