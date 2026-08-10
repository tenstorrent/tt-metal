# ttnn.min example
import torch
import ttnn

device = ttnn.open_device(device_id=0)

torch_input = torch.randn((32, 32), dtype=torch.float32)
ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

ttnn_output = ttnn.mean(
    ttnn_input,
    None,
    False,
    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
)
torch_output = torch.mean(torch_input)

print("Input tensor:", torch_input)
print("TTNN Output:", ttnn_output)
print("Torch Output:", torch_output)

all_close = torch.allclose(ttnn.to_torch(ttnn_output), torch_output, rtol=1e-05, atol=1e-08)
print("Outputs are close:", all_close)

ttnn.close_device(device)
