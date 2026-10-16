# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.all_gather over each axis of a Blackhole galaxy 8x4 torus whose mesh graph descriptor declares
express links (row 2 <-> row 5 chords along the 8-row axis).

Run with
TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/express_links_8x4_mesh_graph_descriptor.textproto
and TT_METAL_OPERATION_TIMEOUT_SECONDS=5, so a hung gather fails instead of blocking.
"""

import os

import pytest
import torch

import ttnn

EXPRESS_MGD = "express_links_8x4_mesh_graph_descriptor.textproto"
MESH_SHAPE = (8, 4)


@pytest.mark.skipif(
    not os.getenv("TT_MESH_GRAPH_DESC_PATH", "").endswith(EXPRESS_MGD),
    reason=f"set TT_MESH_GRAPH_DESC_PATH to the {EXPRESS_MGD} descriptor",
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0_8_rows", "axis1_4_cols"])
def test_all_gather_express_links(mesh_device, cluster_axis):
    ring_size = MESH_SHAPE[cluster_axis]
    torch_input = torch.randn(1, 1, 32, 256 * ring_size).bfloat16()
    shard_dims = (3, None) if cluster_axis == 0 else (None, 3)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=MESH_SHAPE),
    )

    tt_output = ttnn.all_gather(tt_input, dim=3, cluster_axis=cluster_axis, num_links=2, topology=ttnn.Topology.Ring)

    for device_output in ttnn.get_device_tensors(tt_output):
        assert torch.equal(ttnn.to_torch(device_output), torch_input)
