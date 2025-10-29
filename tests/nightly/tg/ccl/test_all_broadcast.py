# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_new_all_broadcast import run_all_broadcast_impl


# Enumerate the post-commit cases explicitly
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, layout, input_dtype",
    [
        (8, 1, [1, 16, 32, 576], ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
        (8, 1, [1, 32, 128, 576], ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
        (8, 1, [1, 32, 32, 576], ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
        (8, 1, [1, 4, 128, 512], ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
        (8, 1, [1, 128, 32, 512], ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
    ],
    ids=["deepseek_1", "deepseek_2", "deepseek_3", "deepseek_4", "deepseek_5"],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1000000}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_broadcast_trace(
    mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")
    if num_devices < 8:
        mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(1, num_devices)
        )
    else:
        mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(-1), ttnn.PlacementReplicate()], ttnn.MeshShape(num_devices, 1)
        )
    run_all_broadcast_impl(
        mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        trace_mode=True,
        mesh_mapper_config=mesh_mapper_config,
    )


@pytest.mark.parametrize("mesh_device", [(8, 16)], indirect=True)
@pytest.mark.parametrize("output_shape", [[8, 16, 128, 576]])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_broadcast_quad_host_mesh(mesh_device, output_shape, num_links, input_dtype, layout):
    import torch

    torch.manual_seed(0)
    input_tensor = torch.rand(output_shape, dtype=torch.bfloat16)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config)
    input_tensor_tt = ttnn.from_torch(
        input_tensor, device=mesh_device, layout=layout, dtype=input_dtype, mesh_mapper=mesh_mapper
    )
    output_tensors = ttnn.experimental.all_broadcast_async(
        input_tensor_tt, num_links=num_links, topology=ttnn.Topology.Linear, cluster_axis=0
    )
    for i, output_tensor in enumerate(output_tensors):
        result_torch = ttnn.to_torch(
            output_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape),
        )
        for j in range(mesh_device.shape[0]):
            assert torch.allclose(result_torch[j, :, :, :], input_tensor[i, :, :, :])
