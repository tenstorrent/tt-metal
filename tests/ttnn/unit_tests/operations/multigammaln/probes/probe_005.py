import torch, ttnn
from ttnn.operations.multigammaln import multigammaln

shape = (1, 1, 32, 32)
torch_input = torch.full(shape, 0.500358, dtype=torch.float32)
device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_output = multigammaln(ttnn_input)
actual = ttnn.to_torch(ttnn_output).float()
print("kernel out[0,0,0,0] =", actual[0, 0, 0, 0].item())
ttnn.close_device(device)
