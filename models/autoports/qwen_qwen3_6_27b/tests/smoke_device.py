# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded by the caller: normal open/add/read/close recovery probe, no reset."""
import torch

import ttnn


def main():
    print("MESH_OPEN_START", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        print("MESH_OPEN_OK", flush=True)
        x = ttnn.from_torch(torch.ones(32, 32), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        y = ttnn.add(x, x)
        actual = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        assert torch.equal(actual, torch.full_like(actual, 2))
        print("DEVICE_ADD_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    print("MESH_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
