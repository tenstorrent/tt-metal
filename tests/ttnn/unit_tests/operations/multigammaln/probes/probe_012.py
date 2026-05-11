import torch, ttnn, math
from ttnn.operations.multigammaln import multigammaln

# a = 1.500358 → for offset=1.5, x_off = 0.000358 (the "problem" value)
shape = (1, 1, 32, 32)
torch_input = torch.full(shape, 1.500358, dtype=torch.float32)
device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out = multigammaln(ttnn_input)
val = ttnn.to_torch(out).float()[0, 0, 0, 0].item()
print(
    f"RESULT D3 (offset=1.5) a=1.500358 (x_off=0.000358): kernel={val} torch.lgamma(0.000358)={math.lgamma(0.000358):.6f}"
)
ttnn.close_device(device)
