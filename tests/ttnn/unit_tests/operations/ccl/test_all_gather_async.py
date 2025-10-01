import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, dtype, memory_config, layout, device):
    torch_input = torch.rand(input_shape).bfloat16()

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )

    return tt_input, torch_reference


MESH_SHAPE = (2, 4)
LAYOUT = ttnn.TILE_LAYOUT

num_iters = 1


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("input_shape", [[1, 1, 1, 32, 32], [2, 2, 2, 32, 32], [2, 2, 32, 32], [5, 32, 32]])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_gather_async(mesh_device, input_shape, dim, cluster_axis, dtype, memory_config, topology):
    if dim >= len(input_shape):
        pytest.skip("Invalid gather dim")

    tt_input, torch_reference = _get_tensors(
        input_shape,
        tuple(mesh_device.shape),
        dim,
        cluster_axis,
        dtype,
        memory_config,
        LAYOUT,
        mesh_device,
    )

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    for i in range(num_iters):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            tt_input,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            multi_device_global_semaphore=semaphores,
        )
    logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        logger.info("Bringing tensor back to host")
        tt_output_tensor = ttnn.to_torch(t)
        logger.info("Brought tensor back from host")

        if dtype == ttnn.bfloat16:
            assert_equal(tt_output_tensor, torch_reference)
        else:
            assert_with_pcc(tt_output_tensor, torch_reference)
