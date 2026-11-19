# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Positive control: a live allocation after capture must block replay."""
import argparse
import json
from pathlib import Path

from ttnn.tools import trace_allocation_tracker

import ttnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-cols", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING, "Tracker is disabled"
    if args.mesh_cols > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh_cols), trace_region_size=0)
    trace = None
    report = {"tracking_enabled": True, "mesh_shape": [1, args.mesh_cols]}
    try:
        x = ttnn.ones((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        warm = ttnn.add(x, x)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = ttnn.add(x, x)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(output)
        unsafe = ttnn.empty((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        report["deliberate_live_buffer"] = unsafe.buffer_address()
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        except RuntimeError as exc:
            report["rejected"] = "device buffer(s) still alive before" in str(exc)
            report["error"] = str(exc)
        else:
            report["rejected"] = False
        assert report["rejected"], "Expected unsafe allocation check did not fire"
        ttnn.deallocate(unsafe)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        report["clean_replay_after_deallocation"] = True
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
