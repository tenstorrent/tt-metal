from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
import torch

torch.manual_seed(0)

device_id = 0
device = ttnn.open_device(device_id=device_id)

ttnn.enable_program_cache(device)

m, n = 4, 4

torch_tensors = [torch.randn((m, n), dtype=torch.bfloat16) for _ in range(3)]
ttnn_tensors = [ttnn.from_torch(tensor) for tensor in torch_tensors]

ttnn_tensors = [ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG) for tensor in ttnn_tensors]

ttnn_tensors = [ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) for tensor in ttnn_tensors]

torch_result = torch.stack(torch_tensors, dim=0)

ttnn_result = ttnn.stack(ttnn_tensors, dim=0)

print("ttnn_result: ", ttnn_result)
print(ttnn_result.shape)
ttnn_result_torch = ttnn.to_torch(ttnn_result)

# assert torch.allclose(torch_result, ttnn_result_torch, atol=1e-2), "TTNN stack result does not match Torch stack result!"
assert_with_pcc(torch_result, ttnn_result_torch)

print("torch_result: ", torch_result)
print(torch_result.shape)
print("Stack operation test passed!")

ttnn.close_device(device)
