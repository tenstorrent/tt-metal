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
    for shape in [(32, 32), (64, 64), (32, 128), (256, 32), (2, 64, 128)]:
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        tx = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
        tg = ttnn.from_torch(
            g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = rms_norm(tx, gamma=tg)
        res = ttnn.to_torch(out).to(torch.float32)
        e = ref(x, g)
        print(shape, "RM PCC=", pcc(res, e), "maxdiff", (res - e).abs().max().item())
finally:
    ttnn.close_device(device)
