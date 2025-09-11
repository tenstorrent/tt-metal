import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    "repeat_factor",
    [
        1,
        2,
        3,
        4,
        5,
    ],
    ids=("1", "2", "3", "4", "5"),
)
def test_tensor_write(mesh_device, repeat_factor):
    print("Num devices: ", mesh_device.get_num_devices())

    tensor_1_shape = [1, 1, 16384, 4]
    tensor_2_shape = [1, 77, 2048]
    tensor_3_shape = [1, 77, 2048]
    tensor_4_shape = [1, 1280]
    tensor_5_shape = [1, 1280]

    torch_tensor_1 = torch.randn(tensor_1_shape)
    torch_tensor_2 = torch.randn(tensor_2_shape)
    torch_tensor_3 = torch.randn(tensor_3_shape)
    torch_tensor_4 = torch.randn(tensor_4_shape)
    torch_tensor_5 = torch.randn(tensor_5_shape)

    ttnn_tensor_1 = ttnn.from_torch(torch_tensor_1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_tensor_2 = ttnn.from_torch(torch_tensor_2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_tensor_3 = ttnn.from_torch(torch_tensor_3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_tensor_4 = ttnn.from_torch(torch_tensor_4, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_tensor_5 = ttnn.from_torch(torch_tensor_5, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn_tensors = [ttnn_tensor_1, ttnn_tensor_2, ttnn_tensor_3, ttnn_tensor_4, ttnn_tensor_5]
    tensors_on_device = []
    for ttnn_tensor in ttnn_tensors:
        tensors_on_device.append(
            ttnn.allocate_tensor_on_device(
                ttnn_tensor.shape, ttnn_tensor.dtype, ttnn_tensor.layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
        )

    num_iters = 100000
    print("Begin tensor loop copy")
    for i in range(num_iters):
        for ttnn_tensor_on_host, tensor_on_device in zip(ttnn_tensors, tensors_on_device):
            ttnn.copy_host_to_device_tensor(ttnn_tensor_on_host, tensor_on_device)
        ttnn.synchronize_device(mesh_device)
        if i % 1000 == 0:
            print(f"Iteration {i} complete")
    print("End tensor loop copy")
