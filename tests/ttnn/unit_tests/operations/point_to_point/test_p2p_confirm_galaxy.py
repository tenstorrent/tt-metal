# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matrix-expansion confirmation: p2p green on a SECOND BH topology — Galaxy 4x2
(mesh (4,2), torus_x), proving the multi-device runner extends beyond bh_8xP150."""
from math import prod
import pytest, torch, ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.point_to_point import point_to_point

@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)  # matches blackhole_8xGalaxy_4x2 (dims [4,2])
def test_p2p_galaxy_4x2(mesh_device):
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("needs >=2 devices")
    sender, receiver = ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)  # same row, adjacent
    n = prod(tuple(mesh_device.shape))
    torch.manual_seed(11)
    full = torch.randn((n, 1, 32, 32), dtype=torch.bfloat16)
    inp = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0))
    ttnn.synchronize_device(mesh_device)
    shards_in = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(inp)]
    out = point_to_point(inp, sender, receiver, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)
    shards_out = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    send_idx = sender[0] * mesh_device.shape[1] + sender[1]   # 0
    recv_idx = receiver[0] * mesh_device.shape[1] + receiver[1]  # 1
    assert_with_pcc(shards_in[send_idx], shards_out[recv_idx], 0.995)
