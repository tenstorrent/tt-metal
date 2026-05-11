import torch, ttnn, math
from ttnn.operations.multigammaln import multigammaln

shape = (1, 1, 32, 32)
torch_input = torch.full(shape, 0.500358, dtype=torch.float32)
device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out = multigammaln(ttnn_input)
val = ttnn.to_torch(out).float()[0, 0, 0, 0].item()
print(f"RESULT D3 (offset=1.5) for a=0.500358: kernel={val} torch.lgamma(-0.999642)={math.lgamma(-0.999642):.6f}")
ttnn.close_device(device)
