# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload an offset distribution through the nonuniform API on a shared mesh.

Launch with tt-run and config/bh_1x4_dual_rank_mesh_graph_descriptor.textproto.
Rank 0 owns columns 0 and 1; rank 1 owns columns 2 and 3. A complete host
tensor occupies columns 1 and 2, so each rank must skip one remote shard.
"""

from pathlib import Path

import pytest
import torch

import ttnn


@pytest.fixture(scope="module")
def co_owned_mesh():
    if not ttnn.using_distributed_env():
        pytest.skip("Requires two ranks that co-own a 1x4 mesh; launch with tt-run")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    try:
        assert int(ttnn.distributed_context_get_size()) == 2
        rank = int(ttnn.distributed_context_get_rank())
        owned = [(coord[0], coord[1]) for coord in mesh.get_view().get_local_mesh_coord_range()]
        assert owned == [(0, 2 * rank), (0, 2 * rank + 1)], owned
        yield mesh, rank
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _populated_coords(tensor):
    buffer = tensor.host_buffer()
    return {(0, col) for col in range(4) if buffer.get_shard(ttnn.MeshCoordinate(0, col)) is not None}


@pytest.mark.parametrize("shard_shape", [(32, 64), (8192, 4096)], ids=["small", "64mib"])
def test_upload_complete_offset_tensor(co_owned_mesh, shard_shape):
    mesh, rank = co_owned_mesh
    if shard_shape == (8192, 4096):
        # 64 MiB per local shard exceeds the 32 MiB pinning threshold; IOMMU enables that path.
        groups = Path("/sys/bus/pci/drivers/tenstorrent").glob("*/iommu_group/type")
        if not any(path.read_text().startswith("DMA") for path in groups):
            pytest.skip("The pinned nonuniform upload regression requires IOMMU")

    source = torch.cat([torch.full(shard_shape, value, dtype=torch.bfloat16) for value in (11, 29)], dim=1)
    mapper = ttnn.create_mesh_mapper(
        ttnn.MeshShape(1, 4),
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)],
            ttnn.MeshShape(1, 2),
            ttnn.MeshCoordinate(0, 1),
        ),
    )
    host = ttnn.from_torch(source, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    # Missing columns 0 and 3 force to_device through non_uniform_data_movement::enqueue_write_tensor.
    assert _populated_coords(host) == {(0, 1), (0, 2)}
    assert [(coord[0], coord[1]) for coord in host.tensor_topology().mesh_coords()] == [(0, 1), (0, 2)]

    device_tensor = ttnn.to_device(host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        readback = ttnn.from_device(device_tensor)
        assert _populated_coords(readback) == {(0, rank + 1)}
        torch.testing.assert_close(
            ttnn.to_torch(readback),
            torch.full(shard_shape, (11, 29)[rank], dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        assert _populated_coords(host) == {(0, 1), (0, 2)}
    finally:
        ttnn.deallocate(device_tensor)
    ttnn.distributed_context_barrier()
