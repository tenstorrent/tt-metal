import ttnn
import torch
import pytest


def check_all_gather(dim, shape, mesh_shape, output_shape, cluster_idx=None):
    multiplier = mesh_shape[0] * mesh_shape[1] if cluster_idx is None else mesh_shape[cluster_idx]
    expected_shape = list(shape)
    expected_shape[dim] *= multiplier
    for i in range(len(expected_shape)):
        if expected_shape[i] != output_shape[i]:
            return False
    return True


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_idx", [1, None])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 1],
        [1, 1, 1, 8],
        [1, 1, 16, 16],
        [1, 1, 32, 32],
        [1, 1, 32, 32],
        [1, 1, 16, 32],
        [1, 1, 4096, 16],
        [1, 1, 4096, 32],
        [1, 1, 4096, 64],
        [1, 1, 4096, 128],
        [1, 1, 4, 2048],
        [1, 1, 8, 2048],
        [1, 1, 16, 2048],
        [1, 1, 32, 2048],
        [1, 1, 64, 2048],
        [1, 1, 2048, 2048],
        [1, 32, 4096, 16],
        [1, 32, 4096, 32],
        [2, 32, 4096, 16],
        [4, 32, 4096, 16],
        [8, 32, 4096, 16],
        [16, 32, 4096, 16],
        [16, 32, 4096, 32],
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 32), id="1x32_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
def test_all_gather_async(dim, cluster_idx, shape, mesh_device):
    print(mesh_device)
    print(mesh_device.shape)

    volume = 1
    for s in shape:
        volume *= s
    torch_tensor = torch.arange(0, volume, dtype=torch.bfloat16).reshape(shape)

    # Shard input activations on batch dimension to devices in the mesh
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

    ccl_sub_device_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))})
    global_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    print("ttnn shape:", ttnn_tensor.shape)

    ttnn_gathered_tensor = None
    if cluster_idx is None:
        ttnn_gathered_tensor = ttnn.experimental.all_gather_async(
            ttnn_tensor, dim, global_semaphores, num_links=1, topology=ttnn.Topology.Linear
        )
    else:
        ttnn_gathered_tensor = ttnn.experimental.all_gather_async(
            ttnn_tensor,
            dim,  # [0, 1, 2, 3]
            cluster_idx,  # 0 or 1
            mesh_device,
            ttnn.Topology.Linear,
            global_semaphores,
        )

    result_shape = list(ttnn_gathered_tensor.shape)
    print(ttnn_gathered_tensor.shape)

    mesh_device_shape = list(mesh_device.shape)

    ttnn.close_mesh_device(mesh_device)
    del mesh_device

    assert check_all_gather(dim, shape, mesh_device_shape, result_shape, cluster_idx), (
        f"All gather failed for dim {dim}, shape {shape}, mesh_shape {mesh_device_shape}, "
        f"result_shape {result_shape}, cluster_idx {cluster_idx}"
    )
