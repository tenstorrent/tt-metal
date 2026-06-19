import torch, ttnn
import ttnn.operations.rms_norm as M

M.SUPPORTED["layout"] = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
M.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]
M.SUPPORTED["gamma_layout"] = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
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


def t(shape, dt=ttnn.bfloat16, gamma=False, glayout=ttnn.TILE_LAYOUT, gdt=ttnn.bfloat16, fp32acc=True):
    x = torch.randn(shape)
    tdt = torch.bfloat16 if dt != ttnn.float32 else torch.float32
    ri = ttnn.from_torch(
        x.to(tdt), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = None
    tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W)
        gtdt = torch.bfloat16 if gdt != ttnn.float32 else torch.float32
        g = ttnn.from_torch(
            tg.to(gtdt).reshape(1, 1, 1, W),
            dtype=gdt,
            layout=glayout,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32acc
    cfg.math_approx_mode = False
    try:
        o = rms_norm(ri, gamma=g, compute_kernel_config=cfg)
        out = ttnn.to_torch(o)
        p = pcc(out, ref(x, tg))
        e = (out.float() - ref(x, tg)).abs().max()
        print(
            f"{shape} dt={str(dt)[-8:]} g={gamma}/{str(glayout)[-4:]}/{str(gdt)[-7:]} fp32acc={fp32acc}: pcc={p:.5f} maxerr={e:.4f} {'OK' if p>0.99 else 'BAD'}"
        )
    except Exception as ex:
        print(f"{shape} dt={dt} g={gamma}: ERR {type(ex).__name__}: {str(ex)[:70]}")


try:
    # non-aligned no gamma
    t((1, 1, 32, 50))
    t((1, 1, 64, 17))
    t((1, 1, 17, 64))
    t((1, 1, 17, 50))
    t((4, 8, 32, 47))
    t((2, 1, 100, 47))
    # gamma TILE
    t((1, 1, 32, 50), gamma=True)
    t((1, 1, 64, 128), gamma=True)
    t((1, 1, 17, 50), gamma=True)
    # gamma RM
    t((1, 1, 32, 50), gamma=True, glayout=ttnn.ROW_MAJOR_LAYOUT)
    t((1, 1, 64, 128), gamma=True, glayout=ttnn.ROW_MAJOR_LAYOUT)
    # fp32
    t((1, 1, 64, 128), dt=ttnn.float32)
    t((1, 1, 32, 50), dt=ttnn.float32, gamma=True, gdt=ttnn.float32, glayout=ttnn.ROW_MAJOR_LAYOUT)
    # multi-block / 3D / 2D
    t((4, 8, 32, 256), gamma=True)
    t((1, 32, 128), gamma=True)
    t((128, 512), gamma=True)
    t((32, 17))
    # wide
    t((1, 1, 32, 4096), gamma=True)
    t((1, 32, 4096), gamma=True)
finally:
    ttnn.close_device(dev)
