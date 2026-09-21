# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Weight-free validation of the FLUX.2 evaluation mesh and runtime."""

import pytest


@pytest.fixture
def device_params():
    import ttnn

    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
def test_replicated_matmul(mesh_device):
    import torch
    import ttnn

    assert mesh_device.arch() == ttnn.Arch.BLACKHOLE
    assert mesh_device.get_num_devices() == 8
    x = ttnn.from_torch(
        torch.ones((1, 1, 32, 32), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    result = ttnn.matmul(x, x)
    ttnn.synchronize_device(mesh_device)
    shards = ttnn.get_device_tensors(result)
    assert len(shards) == 8
    for shard in shards:
        actual = ttnn.to_torch(shard)
        assert torch.equal(actual, torch.full_like(actual, 32))
