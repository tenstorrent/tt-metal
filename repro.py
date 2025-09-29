import pytest
import torch
import ttnn
from tqdm import tqdm

# @pytest.mark.parametrize("mesh_device", [pytest.param((1, 32), id="1x32_grid")], indirect=True)
# @pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
# def test_hang(mesh_device):
#     print(mesh_device)
#     print(mesh_device.shape)

#     shape = [1, 1, 4095, 131071]
#     torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

#     MAX_ITER = 10
#     for it in range(MAX_ITER):
#         # Shard input activations on batch dimension to devices in the mesh
#         print(f"Creating buffer for iteration {it}")

#         ttnn_tensors = []
#         for _ in tqdm(range(5)):
#             with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
#                 print("Run from torch")
#                 ttnn_tensor = ttnn.from_torch(
#                     torch_tensor,
#                     dtype=ttnn.bfloat16,
#                     layout=ttnn.ROW_MAJOR_LAYOUT,
#                     device=mesh_device,
#                 )
#                 print("Run to layout")
#                 ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.TILE_LAYOUT)
#                 print("Done to layout")
#                 # ttnn_tensors.append(ttnn_tensor)
#         print(f"Done for iteration {it}")
#     print("Done")


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 32), id="1x32_grid")], indirect=True)
def test_repro(mesh_device):
    print(mesh_device)
    print(mesh_device.shape)

    shape = [1, 1, 4095, 131071]
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

    MAX_ITER = 10
    for it in range(MAX_ITER):
        print(f"Creating buffer for iteration {it}")

        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            sharded_ttnn_tensor = ttnn.from_torch(
                torch_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
            ttnn_tensor = ttnn.add(sharded_ttnn_tensor, sharded_ttnn_tensor)

    print("Done")
