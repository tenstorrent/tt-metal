import pytest
import torch
import ttnn
from tqdm import tqdm


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 32), id="1x32_grid")], indirect=True)
def test_all_gather_async(mesh_device):
    print(mesh_device)
    print(mesh_device.shape)

    shape = [1, 1, 4096, 2048]
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

    MAX_ITER = 10
    for it in range(MAX_ITER):
        # Shard input activations on batch dimension to devices in the mesh
        print(f"Creating buffer for iteration {it}")

        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            sharded_ttnn_tensor = ttnn.from_torch(
                torch_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
            ttnn_tensor = ttnn.add(sharded_ttnn_tensor, sharded_ttnn_tensor)
            # ttnn.synchronize_device(mesh_device)

    print("Done")
