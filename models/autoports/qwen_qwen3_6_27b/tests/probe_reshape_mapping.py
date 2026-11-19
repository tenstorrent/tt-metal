# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact data, cache ownership, and trace checks for compressed reshape maps."""

import argparse
import json
import math
from pathlib import Path

import torch

import ttnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
    cases = []
    for batch, length, width in [(32, 33, 5120), (32, 34, 2048), (3, 65, 1280), (3, 161, 4352), (3, 17, 64)]:
        pair = ((1, batch * length, width), (batch, length, width))
        cases.extend((pair, pair[::-1]))
    cases.extend([((1, 99, 33), (3, 33, 33)), ((1, 66, 64), (3, 11, 128))])
    results = []
    trace = None
    try:
        for dtype in (ttnn.bfloat16, ttnn.float32, ttnn.uint32):
            for source, target in cases:
                torch_dtype = torch.int32 if dtype == ttnn.uint32 else torch.float32
                host = (torch.arange(math.prod(source), dtype=torch.int32) % 127).reshape(source).to(torch_dtype)
                x = ttnn.from_torch(
                    host,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                before = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank * 8
                output = ttnn.reshape(x, target)
                ttnn.synchronize_device(mesh)
                for local in ttnn.get_device_tensors(output):
                    assert torch.equal(ttnn.to_torch(local).to(torch_dtype), host.reshape(target))
                del local, output
                after = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank * 8
                # The old W5120/S33 unpack map alone occupied22,282,240B.
                if source == (1, 1056, 5120):
                    assert after - before < 1000000, (before, after)
                # New input buffer, same program key, different contents.
                changed = host + 1
                y = ttnn.from_torch(
                    changed,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                warm = ttnn.reshape(y, target)
                del warm
                programs = mesh.num_program_cache_entries()
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                captured = ttnn.reshape(y, target)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for _ in range(2):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                for local in ttnn.get_device_tensors(captured):
                    assert torch.equal(ttnn.to_torch(local).to(torch_dtype), changed.reshape(target))
                assert mesh.num_program_cache_entries() == programs
                ttnn.release_trace(mesh, trace)
                trace = None
                del local, captured, x, y
                results.append(
                    dict(
                        source=source,
                        target=target,
                        dtype=str(dtype),
                        exact_all_ranks=True,
                        retained_map_and_program_bytes=after - before,
                        stable_programs=True,
                    )
                )
                print("RESHAPE_PASS", source, target, dtype, after - before, flush=True)
        args.output.write_text(json.dumps(dict(passed=True, cases=results), indent=2) + "\n")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
