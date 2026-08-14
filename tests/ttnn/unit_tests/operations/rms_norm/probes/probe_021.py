import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.to(torch.float32).reshape(-1)
    return o


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    for name, dt, gdt, fp32acc, shape in [
        ("bf8b accTrue", ttnn.bfloat8_b, ttnn.bfloat16, True, (1, 1, 64, 256)),
        ("bf8b accFalse", ttnn.bfloat8_b, ttnn.bfloat16, False, (1, 1, 64, 256)),
        ("bf8b no gamma", ttnn.bfloat8_b, None, True, (1, 1, 64, 256)),
        ("bf16 + bf8b gamma", ttnn.bfloat16, ttnn.bfloat8_b, True, (1, 1, 64, 256)),
        ("bf8b + bf8b gamma", ttnn.bfloat8_b, ttnn.bfloat8_b, False, (1, 1, 64, 256)),
        ("bf8b + fp32 gamma", ttnn.bfloat8_b, ttnn.float32, True, (1, 1, 64, 256)),
        ("bf8b wide", ttnn.bfloat8_b, ttnn.bfloat16, True, (1, 1, 32, 8192)),
        ("bf8b 2D", ttnn.bfloat8_b, ttnn.bfloat16, True, (1024, 1024)),
        ("bf8b tall", ttnn.bfloat8_b, ttnn.bfloat16, False, (1, 1, 2048, 256)),
        ("fp32 gamma bf8b RM in", ttnn.bfloat16, ttnn.bfloat8_b, True, (1, 1, 64, 256)),
    ]:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        e_g = None
        tg = None
        if gdt is not None:
            g = torch.randn(shape[-1], dtype=torch.bfloat16 if gdt != ttnn.float32 else torch.float32)
            e_g = g
            tg = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=gdt, layout=ttnn.TILE_LAYOUT, device=device)
        e = ref(x, e_g)
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = fp32acc
        cfg.math_approx_mode = False
        try:
            out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg))
            err = (out.float() - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()
            print(f"{name:26s} PCC={pcc(out,e):.6f} relRMS={err:.5f}")
        except Exception as ex:
            print(f"{name:26s} FAIL {type(ex).__name__}: {str(ex)[:200]}")
finally:
    ttnn.close_device(device)
