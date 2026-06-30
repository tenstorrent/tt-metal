# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Throwaway confirmation: the GENERATED point_to_point op runs green on the BH 8xP150
sim when the test's mesh shape matches the runner's topology descriptor (2x4 torus, FABRIC_1D).

The generated acceptance test hardcodes a (1,2)/(1,4) mesh, which does NOT match the
runner's blackhole_8xP150_torus_x mesh-graph descriptor (8 chips, 2x4) -> fabric-init
handshake timeout. This proves the OP is correct; the hang was a test/topology mismatch.
"""
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.point_to_point import point_to_point


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)  # matches blackhole_8xP150 (8 chips)
def test_p2p_confirm(mesh_device):
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("needs >=2 devices")
    sender = ttnn.MeshCoordinate(0, 0)
    receiver = ttnn.MeshCoordinate(0, 1)
    num_devices = prod(tuple(mesh_device.shape))

    torch.manual_seed(42)
    full = torch.randn((num_devices, 1, 32, 32), dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    shards_in = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(inp)]

    out = point_to_point(inp, sender, receiver, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)
    shards_out = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]

    send_idx = sender[0] * mesh_device.shape[1] + sender[1]   # 0
    recv_idx = receiver[0] * mesh_device.shape[1] + receiver[1]  # 1
    assert_with_pcc(shards_in[send_idx], shards_out[recv_idx], 0.995)
