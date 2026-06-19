import torch, ttnn
import ttnn.operations.rms_norm as M

M.SUPPORTED["gamma_layout"] = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
M.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]
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


def t(shape, glayout, gdt=ttnn.bfloat16, dt=ttnn.bfloat16, label=""):
    x = torch.randn(shape)
    W = shape[-1]
    tg = torch.randn(W)
    tdt = torch.bfloat16 if dt != ttnn.float32 else torch.float32
    gtdt = torch.bfloat16 if gdt != ttnn.float32 else torch.float32
    ri = ttnn.from_torch(
        x.to(tdt), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = ttnn.from_torch(
        tg.to(gtdt).reshape(1, 1, 1, W), dtype=gdt, layout=glayout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        out = ttnn.to_torch(rms_norm(ri, gamma=g))
        p = pcc(out, ref(x, tg))
        print(f"{label} TILE-in {shape} g={str(glayout)[-4:]}/{str(gdt)[-7:]}: pcc={p:.5f} {'OK' if p>0.99 else 'BAD'}")
    except Exception as ex:
        print(f"{label} TILE-in {shape} g={str(glayout)[-4:]}: ERR {str(ex)[:60]}")


try:
    # Regime A, RM gamma
    t((1, 1, 64, 128), ttnn.ROW_MAJOR_LAYOUT, label="A")
    t((1, 1, 32, 50), ttnn.ROW_MAJOR_LAYOUT, label="A-Wnonalign")
    t((4, 8, 32, 256), ttnn.ROW_MAJOR_LAYOUT, label="A-multi")
    t((1, 1, 64, 128), ttnn.ROW_MAJOR_LAYOUT, gdt=ttnn.float32, label="A-fp32g")
    # Regime B wide, RM gamma
    t((1, 1, 32, 4096), ttnn.ROW_MAJOR_LAYOUT, label="B-wide")
    t((1, 32, 8192), ttnn.ROW_MAJOR_LAYOUT, label="B-wide2")
    # control: TILE+TILE gamma (non-regression)
    t((1, 1, 64, 128), ttnn.TILE_LAYOUT, label="ctrl-A")
    t((1, 1, 32, 4096), ttnn.TILE_LAYOUT, label="ctrl-B")
finally:
    ttnn.close_device(dev)
