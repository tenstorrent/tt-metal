import torch, ttnn, math
from ttnn.operations.multigammaln import multigammaln

shape = (1, 1, 32, 32)
# Try a=0.000358 fed through offset=0.0 path (which writes 0.000358 to cb_x_off)
torch_input = torch.full(shape, 0.000358, dtype=torch.float32)
device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out = multigammaln(ttnn_input)
val = ttnn.to_torch(out).float()[0, 0, 0, 0].item()
expected = torch.special.multigammaln(torch.tensor(0.000358), 4).item()
print(f"RESULT new-kernel pre-compute a=0.000358: kernel={val} torch.special.multigammaln(0.000358, 4)={expected}")
ttnn.close_device(device)
