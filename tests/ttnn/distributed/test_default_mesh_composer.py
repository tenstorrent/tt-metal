# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""to_torch()/to_numpy() without an explicit mesh_composer on a multi-device tensor.

Repro of record: TT_FATAL at pytensor.cpp ("Can't convert a tensor distributed on
MeshShape([1, 2]) mesh to row-major logical tensor. Supply a mesh composer to
concatenate multi-device shards.") fired from Tensor.to_torch() with
mesh_composer=None.

Part (a): a replicated distribution still raises the original error verbatim --
the fallback must not silently mask distributions it cannot derive.

Part (b): the same site/mesh/layout with a 1-D shard distribution (the exact
case the error message names: "concatenate multi-device shards") now derives a
default composer from the tensor's recorded topology, both host-side and after a
full device write/readback round trip on MeshShape(1, 2).
"""

import pytest
import torch
import ttnn

MESH_SHAPE = (1, 2)  # P300: 2x p150a
SHARD_DIM = 3
SHAPE = (1, 1, 64, 128)  # -> per-device shard (1, 1, 64, 64)


def test_sharded_to_torch_without_composer_on_1x2_mesh():
    device = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
    try:
        torch.manual_seed(0)
        torch_input = torch.randn(*SHAPE, dtype=torch.float32)
        torch_bf16 = torch_input.to(torch.bfloat16)

        # (a) Original failure mode preserved: replicated distribution, no composer.
        replicated = ttnn.from_torch(torch_bf16, mesh_mapper=ttnn.ReplicateTensorToMesh(device))
        with pytest.raises(RuntimeError, match="Supply a mesh composer"):
            replicated.to_torch()

        # (b) Fixed path: 1-D sharded distribution, no composer.
        mapper = ttnn.shard_tensor_to_mesh_mapper(device, dim=SHARD_DIM)
        sharded = ttnn.from_torch(torch_bf16, mesh_mapper=mapper)

        shards = ttnn.get_device_tensors(sharded)
        assert len(shards) == 2
        expected = torch.cat([s.to_torch() for s in shards], dim=SHARD_DIM)

        composed = sharded.to_torch()
        assert tuple(composed.shape) == SHAPE
        assert torch.equal(composed, expected)
        assert torch.equal(composed, torch_bf16)

        # Same site/layout through the device: TILE layout, to_device + from_device.
        sharded_tile = ttnn.from_torch(
            torch_bf16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(device, dim=SHARD_DIM)
        )
        read_back = ttnn.from_device(ttnn.to_device(sharded_tile, device))
        composed_tile = read_back.to_torch()
        assert tuple(composed_tile.shape) == SHAPE
        assert torch.equal(composed_tile, expected)
    finally:
        ttnn.close_mesh_device(device)
