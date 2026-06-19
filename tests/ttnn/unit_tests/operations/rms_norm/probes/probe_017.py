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


def t(shape, dt, gamma=False, gdt=ttnn.bfloat16, ilayout=ttnn.TILE_LAYOUT, glayout=ttnn.TILE_LAYOUT, label=""):
    x = torch.randn(shape)
    ri = ttnn.from_torch(x.bfloat16(), dtype=dt, layout=ilayout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = None
    tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W)
        g = ttnn.from_torch(
            tg.bfloat16().reshape(1, 1, 1, W),
            dtype=gdt,
            layout=glayout,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    try:
        out = ttnn.to_torch(rms_norm(ri, gamma=g, compute_kernel_config=cfg))
        p = pcc(out, ref(x, tg))
        print(f"{label} {shape} idt={str(dt)[-7:]} g={gamma}/{str(gdt)[-7:]}: pcc={p:.5f} {'OK' if p>0.99 else 'BAD'}")
    except Exception as ex:
        print(f"{label} {shape}: ERR {type(ex).__name__}: {str(ex)[:50]}")


try:
    # bf8b input non-aligned (TILE)
    t((128, 100), ttnn.bfloat8_b, label="bf8b-Wnonalign")
    t((1, 1, 17, 64), ttnn.bfloat8_b, label="bf8b-Hnonalign")
    t((1, 1, 17, 50), ttnn.bfloat8_b, label="bf8b-both")
    # bf16 input + bf8b TILE gamma, non-aligned W (gamma W=100 non-aligned)
    t((128, 100), ttnn.bfloat16, gamma=True, gdt=ttnn.bfloat8_b, label="bf16+bf8bg-Wnon")
    # bf8b input aligned + gamma (control)
    t((128, 512), ttnn.bfloat8_b, gamma=True, gdt=ttnn.bfloat8_b, label="bf8b-aligned-ctrl")
    # RM input + bf8b TILE gamma
    t((1, 1, 64, 128), ttnn.bfloat16, gamma=True, gdt=ttnn.bfloat8_b, ilayout=ttnn.ROW_MAJOR_LAYOUT, label="RM+bf8bg")
finally:
    ttnn.close_device(dev)
