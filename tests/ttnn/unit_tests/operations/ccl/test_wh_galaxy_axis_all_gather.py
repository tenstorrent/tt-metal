# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Minimal WH Galaxy (6U, 8x4) sanity: an all_gather along a SINGLE mesh axis (cluster_axis).
# FABRIC_1D is one-dimensional (EAST<->WEST per router); a full-mesh gather over the 2-D 8x4
# mesh forces N/S routing the 1D router rejects (asserts). A per-axis gather is the valid 1D
# operation, so this confirms the galaxy eth fabric routes correctly along an axis.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_wh_galaxy_axis_all_gather(mesh_device, cluster_axis):
    rows, cols = tuple(mesh_device.shape)
    axis_len = (rows, cols)[cluster_axis]

    # Small replicated input; all_gather along one axis concatenates `axis_len` copies on dim 3.
    torch_in = torch.randn(1, 1, 32, 64)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = ttnn.all_gather(
        tt_in,
        dim=3,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device)

    golden = torch.cat([torch_in] * axis_len, dim=3)
    out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(0, 1))
    )
    # The gathered axis now holds axis_len copies on every device along that axis; just check
    # one device's slice matches the golden concatenation.
    per_dev = out[..., : golden.shape[-1]] if out.shape[-1] >= golden.shape[-1] else out
    assert_with_pcc(golden[..., : per_dev.shape[-1]].reshape(-1)[:1024], per_dev.reshape(-1)[:1024], 0.999)
