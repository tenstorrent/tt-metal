# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Complete host tensors on a 1x2 mesh that two ranks co-own, one chip per rank.

A complete host tensor holds every shard of the mesh, including the shard that
belongs to the other rank. Uploading it must write only the local shard.

Launch on one Blackhole host:

    tt-run --mesh-graph-descriptor tests/ttnn/distributed/config/bh_1x2_dual_rank_mesh_graph_descriptor.textproto \\
        --hosts "$(hostname)" pytest tests/ttnn/distributed/test_complete_host_tensor_dual_rank.py

Both ranks must see the same temporary directory. To run the ranks on two
hosts, set TMPDIR to a shared directory and forward it to both ranks.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

MESH_SHAPE = (1, 2)
# Different values per shard make a shard that lands on the wrong chip visible.
SHARD_VALUES = (11, 29)
# The upload takes the pinned-memory path when the IOMMU is on and the local shards exceed this size.
PINNED_WRITE_THRESHOLD_BYTES = 32 * 1024 * 1024


@pytest.fixture(scope="module")
def co_owned_mesh():
    if not ttnn.using_distributed_env():
        pytest.skip("Requires two ranks that co-own a 1x2 mesh; launch with tt-run")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    assert int(ttnn.distributed_context_get_size()) == 2
    rank = int(ttnn.distributed_context_get_rank())
    local_range = mesh_device.get_view().get_local_mesh_coord_range()
    assert [(coord[0], coord[1]) for coord in local_range] == [(0, rank)]
    yield mesh_device, rank
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _populated_coords(host_tensor):
    buffer = host_tensor.host_buffer()
    return {
        coord
        for coord in ((r, c) for r in range(MESH_SHAPE[0]) for c in range(MESH_SHAPE[1]))
        if buffer.get_shard(ttnn.MeshCoordinate(list(coord))) is not None
    }


def _complete_host_tensor(shard_shape):
    torch_tensor = torch.cat([torch.full(shard_shape, value, dtype=torch.bfloat16) for value in SHARD_VALUES], dim=1)
    # A mapper built from a mesh shape has no device, so it constructs every shard on this host.
    mapper = ttnn.create_mesh_mapper(
        ttnn.MeshShape(*MESH_SHAPE),
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]),
    )
    host_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    assert _populated_coords(host_tensor) == {(0, 0), (0, 1)}
    return host_tensor


def _assert_local_shard(device_tensor, rank, shard_shape):
    host_tensor = ttnn.from_device(device_tensor)
    assert _populated_coords(host_tensor) == {(0, rank)}
    expected = torch.full(shard_shape, SHARD_VALUES[rank], dtype=torch.bfloat16)
    torch.testing.assert_close(ttnn.to_torch(host_tensor), expected, rtol=0, atol=0)


def _iommu_enabled():
    # UMD reads the same sysfs attribute and requires one IOMMU state across all devices.
    group_types = Path("/sys/bus/pci/drivers/tenstorrent").glob("*/iommu_group/type")
    return any(path.read_text().startswith("DMA") for path in group_types)


@pytest.mark.parametrize("shard_shape", [(32, 64), (8192, 4096)], ids=["small", "64mib"])
def test_upload_complete_host_tensor(co_owned_mesh, shard_shape):
    mesh_device, rank = co_owned_mesh
    host_tensor = _complete_host_tensor(shard_shape)

    local_bytes = shard_shape[0] * shard_shape[1] * 2
    pinned = _iommu_enabled() and local_bytes > PINNED_WRITE_THRESHOLD_BYTES
    # try_pin can still fall back to an unpinned write, and nothing reports it, so record only the selection.
    logger.info(f"rank {rank}: {local_bytes} local bytes, pinned write path selected: {pinned}")

    device_tensor = ttnn.to_device(host_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _assert_local_shard(device_tensor, rank, shard_shape)
    ttnn.deallocate(device_tensor)
    # The upload must not consume or trim the host shards.
    assert _populated_coords(host_tensor) == {(0, 0), (0, 1)}
    ttnn.distributed_context_barrier()


def test_load_complete_file(co_owned_mesh):
    mesh_device, rank = co_owned_mesh
    shard_shape = (32, 64)
    path = Path(tempfile.gettempdir()) / f"tt-metal-complete-host-tensor-{os.getuid()}.tensorbin"
    if rank == 0:
        ttnn.dump_tensor(path, _complete_host_tensor(shard_shape), mode=ttnn.DumpTensorMode.LOCAL)
    ttnn.distributed_context_barrier()

    # Rank 1 reads a file that rank 0 wrote, with both shards in it.
    assert _populated_coords(ttnn.load_tensor(path)) == {(0, 0), (0, 1)}
    device_tensor = ttnn.load_tensor(path, device=mesh_device)
    _assert_local_shard(device_tensor, rank, shard_shape)
    ttnn.deallocate(device_tensor)

    # Rank 0 must not remove the file before rank 1 has read it.
    ttnn.distributed_context_barrier()
    if rank == 0:
        path.unlink()
