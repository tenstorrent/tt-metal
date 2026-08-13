import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return o * g.to(torch.float32).reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(42)
    for shape in [(32, 50), (32, 17), (17, 64), (47, 100), (2, 4, 47, 47), (32, 40), (32, 72)]:
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
        for lay, name in [(ttnn.TILE_LAYOUT, "TILE"), (ttnn.ROW_MAJOR_LAYOUT, "RM")]:
            tx = ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tg = ttnn.from_torch(
                g,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            res = ttnn.to_torch(rms_norm(tx, gamma=tg)).to(torch.float32)
            e = ref(x, g)
            print(name, shape, "PCC=", round(pcc(res, e), 6), "maxdiff", round((res - e).abs().max().item(), 4))
finally:
    ttnn.close_device(device)
