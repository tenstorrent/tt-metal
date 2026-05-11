import torch, ttnn
from ttnn.operations.multigammaln import multigammaln

torch.manual_seed(42)
# Use the EXACT problematic inputs from the probe
device = ttnn.open_device(device_id=0)

# Place 0.500358 in a single-tile input to isolate
shape = (1, 1, 32, 32)
torch_input = torch.full(shape, 0.500358, dtype=torch.float32)

# Math by hand using torch (each lgamma term + constant):
import math

a = torch_input[0, 0, 0, 0].item()
print(f"a = {a:.10f}")
print(f"  lgamma(a-0.0) = {math.lgamma(a):.6f}")
print(f"  lgamma(a-0.5) = {math.lgamma(a-0.5):.6f}")
print(f"  lgamma(a-1.0) = {math.lgamma(a-1.0):.6f}")
print(f"  lgamma(a-1.5) = {math.lgamma(a-1.5):.6f}")
print(f"  +3*log(pi)    = {3*math.log(math.pi):.6f}")
print(
    f"  sum           = {math.lgamma(a) + math.lgamma(a-0.5) + math.lgamma(a-1.0) + math.lgamma(a-1.5) + 3*math.log(math.pi):.6f}"
)
print(f"  torch.special.multigammaln(a, 4) = {torch.special.multigammaln(torch.tensor(a), 4).item():.6f}")

ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_output = multigammaln(ttnn_input)
actual = ttnn.to_torch(ttnn_output).float()
print(f"  kernel output[0,0,0,0] = {actual[0,0,0,0].item()}")
ttnn.close_device(device)
