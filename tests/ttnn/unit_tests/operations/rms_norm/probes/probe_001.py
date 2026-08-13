import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

torch.manual_seed(0)
shape = (32, 32)
x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
tx = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out = rms_norm(tx)
res = ttnn.to_torch(out).to(torch.float32)
xf = x.to(torch.float32)
exp = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
print("max diff", (res - exp).abs().max().item())
print(res[0, :8])
print(exp[0, :8])
