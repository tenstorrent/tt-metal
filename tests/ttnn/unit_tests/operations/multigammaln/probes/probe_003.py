import torch, ttnn

device = ttnn.open_device(device_id=0)

# Helper: copy x into all four cb_lgamma_* manually using ttnn ops
# Instead, let's just isolate by editing kernel separately. Here probe one offset at a time.
# Simpler: feed a = 0.500358 + offset to test each lgamma in isolation as offset=0
# (i.e., feed b = a - offset for each offset, all with offset=0 path).

import math

a = 0.500358
for offset in [0.0, 0.5, 1.0, 1.5]:
    b = a - offset
    # What torch.lgamma produces for input b:
    print(f"offset={offset}: x_off={b:.6f}, torch.lgamma(x_off)={math.lgamma(b):.6f}")

# Test ttnn.lgamma if available
torch_input = torch.tensor([[[[0.500358 - 0.5]]]], dtype=torch.float32).repeat(1, 1, 32, 32)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
try:
    out = ttnn.lgamma(ttnn_input)
    val = ttnn.to_torch(out).float()[0, 0, 0, 0].item()
    print(f"ttnn.lgamma(0.000358) = {val}")
except Exception as e:
    print(f"ttnn.lgamma failed: {e}")
ttnn.close_device(device)
