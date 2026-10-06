# SPDX-License-Identifier: Apache-2.0
"""Native selection semantics for skipped inactive attention rows."""

import json
from pathlib import Path

import torch

import ttnn


def main():
    torch.manual_seed(42)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        host = torch.randn(1, 1, 32, 1536, dtype=torch.bfloat16)
        host[..., -1, :] = float("nan")
        values = list(range(31)) + [-1]
        values[0] = 1048575
        positions = ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        y = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        tiled = ttnn.to_layout(ttnn.reshape(positions, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        active = ttnn.transpose(ttnn.typecast(ttnn.gez(tiled), ttnn.bfloat16), -2, -1)
        selected = ttnn.where(active, y, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ranks = []
        for rank, tensor in enumerate(ttnn.get_device_tensors(selected)):
            result = ttnn.to_torch(tensor)
            assert torch.equal(result[..., :-1, :], host[..., :-1, :])
            assert torch.count_nonzero(result[..., -1, :]).item() == 0
            assert torch.isfinite(result).all()
            ranks.append(dict(rank=rank, active_exact=True, inactive_nan_selected_to_zero=True))
        path = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/lifetime_fix/mask_api_probe.json"
        path.write_text(json.dumps(dict(status="pass", positions=values, ranks=ranks), indent=2) + "\n")
        print("MASK_API_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
