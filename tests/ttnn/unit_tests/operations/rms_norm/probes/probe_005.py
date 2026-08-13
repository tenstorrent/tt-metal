import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    x = torch.randn((32, 32), dtype=torch.float32).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(tx)
    res = ttnn.to_torch(out).to(torch.float32)
    xf = x.to(torch.float32)
    exp = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    print("max diff", (res - exp).abs().max().item())
    print("ok")
finally:
    ttnn.close_device(device)
