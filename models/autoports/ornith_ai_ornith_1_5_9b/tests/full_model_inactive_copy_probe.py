# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise inactive-copy controls at B1/B3/B32, tracked repeated mask changes."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.inactive_state_copy import inactive_state_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
    result = {"status": "running", "cases": []}
    trace = None
    try:
        for batch in (1, 3, 32):
            original = torch.randn(batch, 8, 128, 128, generator=torch.Generator().manual_seed(batch))
            changed = original * 1.125 + 0.5

            def upload(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=True):
                return ttnn.from_torch(
                    value,
                    device=mesh if device else None,
                    dtype=dtype,
                    layout=layout,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )

            old, new = upload(original), upload(changed)
            pool, saved = ttnn.zeros_like(old), ttnn.zeros_like(old)
            active = upload(torch.ones((1, 32), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

            def operation(pool=pool, saved=saved, active=active, new=new):
                inactive_state_copy(pool, saved, active)
                ttnn.copy(new, pool)
                inactive_state_copy(saved, pool, active)

            ttnn.copy(old, pool)
            operation()
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            operation()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            programs = mesh.num_program_cache_entries()
            mesh.set_program_cache_misses_allowed(False)
            masks = [
                [1] * batch,
                [0] * batch,
                [int(i != 0) for i in range(batch)],
                [int(i != batch - 1) for i in range(batch)],
                [i % 2 for i in range(batch)],
            ]
            for mask in masks + masks[::-1]:
                bits = torch.ones((1, 32), dtype=torch.int32)
                bits[0, :batch] = torch.tensor(mask, dtype=torch.int32)
                host = upload(bits, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False)
                ttnn.copy_host_to_device_tensor(host, active)
                ttnn.copy(old, pool)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                expected = torch.where(torch.tensor(mask, dtype=torch.bool)[:, None, None, None], changed, original)
                for rank, shard in enumerate(ttnn.get_device_tensors(pool)):
                    actual = ttnn.to_torch(shard)
                    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (batch, mask, rank)
                assert mesh.num_program_cache_entries() == programs
                result["cases"].append({"batch": batch, "active": mask, "ranks_bit_equal": 4})
            ttnn.release_trace(mesh, trace)
            trace = None
            mesh.set_program_cache_misses_allowed(True)
            del old, new, pool, saved, active
        result["status"] = "pass"
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
