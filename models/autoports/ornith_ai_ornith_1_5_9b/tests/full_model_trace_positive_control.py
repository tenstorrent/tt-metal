# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prove the enabled trace allocator rejects a deliberately live late buffer."""

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn


def main():
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=8 * 1024 * 1024)
    trace = None
    try:
        value = ttnn.from_torch(
            torch.ones((1, 1, 32, 32)),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        warm = ttnn.add(value, value)
        del warm
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = ttnn.add(value, value)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        del output
        late = ttnn.empty_like(value)
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        except RuntimeError as error:
            assert "buffer" in str(error) and "alive" in str(error), str(error)
            print("POSITIVE_CONTROL_REJECTED", str(error), flush=True)
        else:
            raise AssertionError("Allocation tracker accepted the deliberately live late buffer")
        del late
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        print("POSITIVE_CONTROL_PASS: replay succeeds after releasing the unsafe buffer", flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
