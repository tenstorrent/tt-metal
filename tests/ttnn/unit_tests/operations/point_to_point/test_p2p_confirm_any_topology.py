# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Topology-ADAPTIVE p2p confirmation — one test that runs on ANY topology in the matrix.

The other confirmation tests hardcode a mesh shape (2,4 / 4,2), so each matches only one
topology; running them under a different topology hangs fabric init. THIS test instead reads
the topology's mesh shape + fabric_config from the environment (set per-topology by
scripts/run_multidevice_sim_pytest.py), so a SINGLE command fans p2p verification out across
EVERY topology whose applies_to_ops lists point_to_point:

    scripts/run_multidevice_sim_pytest.py --op point_to_point -- \
        tests/ttnn/unit_tests/operations/point_to_point/test_p2p_confirm_any_topology.py

The runner exports MULTIDEV_SIM_MESH_SHAPE (e.g. "8,4") + MULTIDEV_SIM_FABRIC_CONFIG
(e.g. "FABRIC_1D") for the active topology; this module reads them at collection time to
parametrize the mesh_device + device_params fixtures. Default (2,4)/FABRIC_1D when run
outside the runner. The oracle is p2p's identity: receiver out-shard == sender in-shard.
"""
import os
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.point_to_point import point_to_point

# Read the active topology from the env the multichip runner sets (fallback = bh_8xP150).
_MESH = tuple(int(x) for x in os.environ.get("MULTIDEV_SIM_MESH_SHAPE", "2,4").split(","))
_FABRIC = getattr(ttnn.FabricConfig, os.environ.get("MULTIDEV_SIM_FABRIC_CONFIG", "FABRIC_1D"))
_TOPO = os.environ.get("MULTIDEV_SIM_TOPOLOGY", "<default>")


@pytest.mark.parametrize("device_params", [{"fabric_config": _FABRIC}], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_p2p_any_topology(mesh_device):
    """p2p identity on whatever topology the runner selected (mesh shape from env)."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("needs >=2 devices")
    print(f"[p2p-any] topology={_TOPO} mesh={tuple(mesh_device.shape)} fabric={_FABRIC}")
    sender, receiver = ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)  # adjacent, row 0
    n = prod(tuple(mesh_device.shape))
    torch.manual_seed(7)
    full = torch.randn((n, 1, 32, 32), dtype=torch.bfloat16)
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
    cols = mesh_device.shape[1]
    send_idx = sender[0] * cols + sender[1]
    recv_idx = receiver[0] * cols + receiver[1]
    assert_with_pcc(shards_in[send_idx], shards_out[recv_idx], 0.999)
