# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "dtype,torch_dtype,layout",
    [
        (ttnn.uint32, torch.int32, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.bfloat16, torch.bfloat16, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("write", ["single_host_view", "mesh_host_view", "whole_mesh"])
@pytest.mark.parametrize("memory", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_copy_host_to_device_preserves_other_views(mesh_device, dtype, torch_dtype, layout, replicated, write, memory):
    shape = (1, 1, 32, 32)
    inputs = [torch.full(shape, 10 + rank, dtype=torch_dtype) for rank in range(2)]
    zeros = torch.zeros(shape if replicated else (2, 1, 32, 32), dtype=torch_dtype)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if replicated else ttnn.ShardTensorToMesh(mesh_device, dim=0)
    target = ttnn.from_torch(
        zeros, dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=mapper, memory_config=memory
    )
    views = ttnn.get_device_tensors(target)
    expected = [torch.zeros_like(value) for value in inputs]
    mesh_host = ttnn.from_torch(
        torch.cat(inputs, 0), dtype=dtype, layout=layout, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    def check_values():
        for rank, (held, fresh) in enumerate(zip(views, ttnn.get_device_tensors(target))):
            for view in (held, fresh):
                actual = ttnn.to_torch(view)
                assert tuple(actual.shape) == shape
                assert torch.equal(actual.to(torch.int64), expected[rank].to(torch.int64))

    if write == "whole_mesh":
        ttnn.copy_host_to_device_tensor(mesh_host, target)
        expected = inputs
        check_values()
    else:
        for rank, view in enumerate(views):
            host = mesh_host if write == "mesh_host_view" else ttnn.from_torch(inputs[rank], dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host, view)
            expected[rank] = inputs[rank]
            check_values()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("host_width", [2, 8, 1], ids=["smaller_host", "larger_host", "replicated_host"])
@pytest.mark.parametrize("view_ranks", [(0, 2), (2,), (2, 3)], ids=["mixed_view", "single_view", "absent_view"])
def test_copy_host_to_device_view_with_different_source_mesh(mesh_device, host_width, view_ranks):
    shape = (1, 1, 32, 32)
    initial = [torch.full(shape, 100 + rank, dtype=torch.int32) for rank in range(4)]
    target = ttnn.from_torch(
        torch.cat(initial, dim=0),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    views = ttnn.get_device_tensors(target)
    destination = (
        views[view_ranks[0]]
        if len(view_ranks) == 1
        else ttnn.combine_device_tensors(tensors=[views[rank] for rank in view_ranks])
    )
    destination_coords = [ttnn.MeshCoordinate(0, rank) for rank in view_ranks]
    assert destination.device_coords() == destination_coords

    inputs = [torch.full(shape, 10 + rank, dtype=torch.int32) for rank in range(host_width)]
    host = ttnn.from_host_shards(
        [ttnn.from_torch(value, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for value in inputs],
        ttnn.MeshShape(1, host_width),
    )
    ttnn.copy_host_to_device_tensor(host, destination)

    assert destination.device_coords() == destination_coords
    expected = list(initial)
    for rank in view_ranks:
        if host_width == 1:
            expected[rank] = inputs[0]
        elif rank < host_width:
            expected[rank] = inputs[rank]
    for rank, (held, fresh) in enumerate(zip(views, ttnn.get_device_tensors(target))):
        for view in (held, fresh):
            assert torch.equal(ttnn.to_torch(view).to(torch.int64), expected[rank].to(torch.int64))


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("view_ranks", [(2,), (0, 1, 2)], ids=["unpopulated_view", "mixed_view"])
def test_copy_host_to_device_view_skips_unpopulated_source_shards(mesh_device, view_ranks):
    shape = (1, 1, 32, 32)
    initial = [torch.full(shape, 100 + rank, dtype=torch.int32) for rank in range(4)]
    target = ttnn.from_torch(
        torch.cat(initial, dim=0),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    views = ttnn.get_device_tensors(target)
    destination = (
        views[view_ranks[0]]
        if len(view_ranks) == 1
        else ttnn.combine_device_tensors(tensors=[views[rank] for rank in view_ranks])
    )
    destination_coords = [ttnn.MeshCoordinate(rank // 2, rank % 2) for rank in view_ranks]
    assert destination.device_coords() == destination_coords

    inputs = [torch.full(shape, 10 + rank, dtype=torch.int32) for rank in range(2)]
    host = ttnn.from_torch(
        torch.cat(inputs, dim=0),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(1, 2), dims=(None, 0)),
    )
    assert len(ttnn.get_device_tensors(host)) == 2
    ttnn.copy_host_to_device_tensor(host, destination)

    assert destination.device_coords() == destination_coords
    expected = list(initial)
    for rank in view_ranks:
        if rank < 2:
            expected[rank] = inputs[rank]
    for rank, (held, fresh) in enumerate(zip(views, ttnn.get_device_tensors(target))):
        for view in (held, fresh):
            assert torch.equal(ttnn.to_torch(view).to(torch.int64), expected[rank].to(torch.int64))
