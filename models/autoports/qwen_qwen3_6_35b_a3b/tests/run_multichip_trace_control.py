# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prove allocation tracking rejects a live unsafe buffer on the target mesh."""

import argparse
import gc
import json
import os
from pathlib import Path

# Configuration must be supplied by the launcher before importing TTNN.
assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
assert "TT_METAL_WATCHER" not in os.environ, "Unset watcher; a value of 0 enables zero-interval polling"

import torch
from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance.update(device="Blackhole P300c 1x4 target mesh", decoder="multichip trace allocation control")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print("PROVENANCE", json.dumps(provenance), flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000)
    print("MESH_OPENED", flush=True)
    trace = None
    try:
        host_input = torch.arange(1024, dtype=torch.bfloat16).reshape(1, 1, 32, 32)
        trace_input = ttnn.from_torch(
            host_input,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        warm = ttnn.neg(trace_input)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        print("WARMUP_COMPLETE", flush=True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = ttnn.neg(trace_input)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        # Only this capture output is acknowledged and kept alive through release.
        acknowledge_corruptible(trace_output)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        unsafe = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh)
        unsafe_id = unsafe.buffer_unique_id()
        assert unsafe_id in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        except RuntimeError as error:
            exception = str(error)
            assert "device buffer(s) still alive before" in exception and f"Buffer {unsafe_id}" in exception
            print("EXPECTED_REJECTION", exception, flush=True)
        else:
            raise AssertionError("Unsafe live allocation did not reject trace replay")
        ttnn.deallocate(unsafe)
        del unsafe
        gc.collect()
        assert unsafe_id not in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        replicas = [ttnn.to_torch(part) for part in ttnn.get_device_tensors(trace_output)]
        assert len(replicas) == 4 and all(torch.equal(part, -host_input) for part in replicas)
        result = dict(
            passed=True,
            mesh=[1, 4],
            fabric="FABRIC_1D",
            rejected=True,
            exception_type="RuntimeError",
            exception=exception,
            unsafe_buffer_id=unsafe_id,
            acknowledged_output_buffer_id=trace_output.buffer_unique_id(),
            unsafe_removed=True,
            clean_replay_passed=True,
            exact_output_replicas=4,
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("TRACE_ALLOCATION_POSITIVE_CONTROL PASS", json.dumps(result), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
