import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    xf = x.float()
    rms = torch.sqrt((xf**2).mean(-1, keepdim=True) + eps)
    o = xf / rms
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


dev = ttnn.open_device(device_id=0)


def t(shape, glayout):
    x = torch.randn(shape)
    W = shape[-1]
    tg = torch.randn(W)
    ri = ttnn.from_torch(
        x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = ttnn.from_torch(
        tg.bfloat16().reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=glayout,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        out = ttnn.to_torch(rms_norm(ri, gamma=g))
        p = pcc(out, ref(x, tg))
        print(f"TILE-in {shape} gamma={str(glayout)[-4:]}: pcc={p:.5f} {'OK' if p>0.99 else 'BAD'}")
    except Exception as ex:
        print(f"TILE-in {shape} gamma={str(glayout)[-4:]}: ERR {str(ex)[:60]}")


try:
    t((1, 1, 64, 128), ttnn.ROW_MAJOR_LAYOUT)  # TILE input + RM gamma (Regime A)
    t((1, 1, 32, 4096), ttnn.ROW_MAJOR_LAYOUT)  # TILE input + RM gamma (Regime B wide)
    t((1, 1, 64, 128), ttnn.TILE_LAYOUT)  # control: TILE+TILE gamma (should pass)
finally:
    ttnn.close_device(dev)
