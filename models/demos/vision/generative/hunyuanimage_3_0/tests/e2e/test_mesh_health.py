# SPDX-License-Identifier: Apache-2.0
"""Phase-0b mesh/fabric health probe. Opening the mesh under FABRIC_1D IS the
fabric-init test (the (1,1)+FABRIC_1D open previously failed at fabric router sync).
If this passes setup + a trivial CCL all_gather, the mesh HW-parallel (÷8) path is live.
"""
from __future__ import annotations

import pytest
import torch

import ttnn

try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (8, 4)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_mesh_health(device_params, mesh_device):
    nd = mesh_device.get_num_devices()
    print(f"\nMESH_OPEN_OK shape={tuple(mesh_device.shape)} ndev={nd}", flush=True)

    # trivial per-device op
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 32 * nd),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    y = ttnn.add(x, x)
    ttnn.synchronize_device(mesh_device)
    print("MESH_LOCAL_OP_OK", flush=True)

    # exercise the fabric with a CCL all_gather across the mesh
    try:
        g = ttnn.all_gather(x, dim=3, num_links=1)
        ttnn.synchronize_device(mesh_device)
        print(f"MESH_ALLGATHER_OK out_shape={tuple(g.shape)}", flush=True)
    except Exception as e:
        print(f"MESH_ALLGATHER_FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
