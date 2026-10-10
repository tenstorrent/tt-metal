# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""all_to_all_async_generic on a mesh view whose axes run against the fabric mesh's (a 4x2 view of the 2x4 p150 x8
fabric: logical axis 0 is the fabric's East-West). The kernel's multicast initialization routes axis 1 as East-West
and axis 0 as North-South, so on such a view it went to the wrong chips and the init barrier hung (both axes); the
host now takes the per-target unicast initialization there. 2x4 is the aligned control. Needs 8 Blackhole chips."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 2), (2, 4)], indirect=True, ids=["4x2", "2x4"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("topo", ["default", "Linear"])
def test_all_to_all_view_orientation(mesh_device, axis, topo):
    rows_mesh, cols_mesh = tuple(mesh_device.shape)
    n = mesh_device.shape[axis]
    rows = 64
    x = torch.randn(1, 8 * n, rows * n, 512).bfloat16()
    dims = (None, 1) if axis == 1 else (1, None)
    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    kw = {} if topo == "default" else {"topology": getattr(ttnn.Topology, topo)}
    y = ttnn.experimental.all_to_all_async_generic(
        t, in_dim=1, out_dim=2, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=axis, **kw
    )
    ttnn.synchronize_device(mesh_device)
    parts = [ttnn.to_torch(p) for p in ttnn.get_device_tensors(y)]
    for d, p in enumerate(parts):
        i = (d % cols_mesh) if axis == 1 else (d // cols_mesh)
        assert torch.equal(p, x[:, :, i * rows : (i + 1) * rows]), f"chip {d} wrong"
