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
