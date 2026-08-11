import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def ref(x, g=None, eps=1e-6):
    xf = x.float()
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


device = ttnn.open_device(device_id=0)
try:
    print("tile_size bfp8:", ttnn.tile_size(ttnn.bfloat8_b))
    for name, dtype, fp32acc in [
        ("bf16/accFalse", ttnn.bfloat16, False),
        ("bf16/accTrue", ttnn.bfloat16, True),
        ("bfp8/accFalse", ttnn.bfloat8_b, False),
        ("bfp8/accTrue", ttnn.bfloat8_b, True),
    ]:
        for shape in [(1, 1, 32, 64), (1, 1, 64, 128), (1, 1, 32, 4096), (1, 1, 32, 8192), (2, 4, 128, 512)]:
            torch.manual_seed(0)
            x = torch.randn(shape, dtype=torch.bfloat16)
            g = torch.randn(shape[-1], dtype=torch.bfloat16)
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi4
            cfg.fp32_dest_acc_en = fp32acc
            cfg.math_approx_mode = False
            try:
                ti = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
                tg = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
                print("  elem_size", ti.element_size())
                out = rms_norm(ti, gamma=tg, compute_kernel_config=cfg)
                got = ttnn.to_torch(out)
                exp = ref(x, g)
                p = pcc(got, exp)
                err = (got.float() - exp).pow(2).mean().sqrt() / exp.std()
                print(f"  {name} {shape}: PCC={p:.6f} rms_rel={err:.4f}")
            except Exception as e:
                print(f"  {name} {shape}: EXC {type(e).__name__}: {str(e)[:300]}")
finally:
    ttnn.close_device(device)
