# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Focused CCL microbenchmarks for the exact GLM sparse-MLA tensor shapes.

Run individual nodes through ``scripts/run_safe_pytest.sh --profile`` to collect
device-kernel timings.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal

LOUD_BOX_MESH = (2, 4)
FABRIC_2D_DEVICE_PARAMS = {"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_2D}


def _global_semaphores(mesh_device):
    compute_grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    gather_semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)
    return gather_semaphores, barrier_semaphore


@pytest.mark.parametrize("mesh_device", [LOUD_BOX_MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC_2D_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "local_tokens,check_values",
    [pytest.param(7040, True, id="warm"), pytest.param(64640, False, id="long")],
)
def test_kvpe_all_gather_perf(mesh_device, local_tokens, check_values):
    """Profile the SP all-gather used for the GLM KVPE prefix."""
    global_shape = [1, 1, local_tokens * 2, 576]
    torch_input = (
        torch.rand(global_shape, dtype=torch.bfloat16)
        if check_values
        else torch.zeros(global_shape, dtype=torch.bfloat16)
    )
    mesh_mapper = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
        mesh_device.shape,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper),
    )
    gather_semaphores, barrier_semaphore = _global_semaphores(mesh_device)

    for iteration in range(2):
        tt_output = ttnn.experimental.all_gather_async(
            tt_input,
            dim=2,
            multi_device_global_semaphore=gather_semaphores,
            barrier_semaphore=barrier_semaphore,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=0,
        )
        ttnn.synchronize_device(mesh_device)
        assert list(tt_output.shape) == global_shape
        assert tt_output.tensor_topology().placements() == [
            ttnn.PlacementReplicate(),
            ttnn.PlacementReplicate(),
        ]
        if check_values and iteration == 1:
            for device_tensor in ttnn.get_device_tensors(tt_output):
                equal, message = comp_equal(torch_input, ttnn.to_torch(device_tensor))
                assert equal, message
        ttnn.deallocate(tt_output)


def _run_glm_all_to_all(mesh_device, logical_shape, in_dim, out_dim):
    torch_input = torch.rand(logical_shape, dtype=torch.bfloat16)
    mesh_mapper = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(out_dim), ttnn.PlacementShard(in_dim)],
        mesh_device.shape,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper),
    )

    for iteration in range(2):
        tt_output = ttnn.experimental.all_to_all_async_2d(
            tt_input,
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=1,
        )
        ttnn.synchronize_device(mesh_device)
        if iteration == 1:
            actual = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=out_dim))
            equal, message = comp_equal(torch_input, actual)
            assert equal, message
        ttnn.deallocate(tt_output)


@pytest.mark.parametrize("mesh_device", [LOUD_BOX_MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC_2D_DEVICE_PARAMS], indirect=True)
def test_glm_head_to_sequence_all_to_all_perf(mesh_device):
    """Profile ``[1,16,640,576] -> [1,64,160,576]`` across TP=4."""
    _run_glm_all_to_all(mesh_device, [1, 64, 1280, 576], in_dim=1, out_dim=2)


@pytest.mark.parametrize("mesh_device", [LOUD_BOX_MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC_2D_DEVICE_PARAMS], indirect=True)
def test_glm_sequence_to_head_all_to_all_perf(mesh_device):
    """Profile ``[1,64,160,512] -> [1,16,640,512]`` across TP=4."""
    _run_glm_all_to_all(mesh_device, [1, 128, 640, 512], in_dim=2, out_dim=1)
