# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small TP4 positive control for the serving allocation-tracker environment."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=2 * 1024 * 1024)
    trace = None
    result = {"status": "running", "mesh": [1, 4], "program_cache_included": True}
    try:
        value = ttnn.from_torch(
            torch.ones((1, 1, 32, 32)),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        warm = ttnn.add(value, value)
        del warm
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = ttnn.add(value, value)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        # Sole trace owns and overwrites this output; no other consumer persists.
        trace_allocation_tracker.acknowledge_corruptible(output)
        unsafe = ttnn.empty((1, 1, 32, 32), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        except RuntimeError as error:
            assert "alloc" in str(error).lower() or "unsafe" in str(error).lower()
            result["positive_control_error"] = str(error)
        else:
            raise AssertionError("Tracker did not reject a live unacknowledged allocation")
        del unsafe
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        assert all(
            torch.equal(ttnn.to_torch(shard), torch.full((1, 1, 32, 32), 2.0))
            for shard in ttnn.get_device_tensors(output)
        )
        result.update(status="pass", replay_after_release=True, all_four_shards_correct=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
