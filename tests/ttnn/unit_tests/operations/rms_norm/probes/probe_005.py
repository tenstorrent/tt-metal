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
    # can ttnn even build a ROW_MAJOR bfp8 tensor?
    try:
        t = ttnn.from_torch(
            torch.randn(32, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        print("rm_bfp8: BUILT", t.layout, t.dtype)
    except Exception as e:
        print("rm_bfp8: refused ->", type(e).__name__, str(e)[:200])

    for name, dtype, gdtype, fp32acc in [
        ("bfp8-in/bfp8-g/accT", ttnn.bfloat8_b, ttnn.bfloat8_b, True),
        ("bfp8-in/bfp8-g/accF", ttnn.bfloat8_b, ttnn.bfloat8_b, False),
        ("bf16-in/bfp8-g/accT", ttnn.bfloat16, ttnn.bfloat8_b, True),
        ("bf16-in/bfp8-g/accF", ttnn.bfloat16, ttnn.bfloat8_b, False),
        ("fp32-in/bfp8-g/accT", ttnn.float32, ttnn.bfloat8_b, True),
        ("bfp8-in/fp32-g/accT", ttnn.bfloat8_b, ttnn.float32, True),
        ("bfp8-in/no-g/accT", ttnn.bfloat8_b, None, True),
    ]:
        print("== " + name)
        for shape in [
            (1, 1, 32, 64),
            (1, 1, 64, 128),
            (4, 8, 32, 256),
            (1, 1, 32, 4096),
            (1, 1, 32, 8192),
            (2, 4, 128, 512),
            (1024, 1024),
        ]:
            torch.manual_seed(0)
            x = torch.randn(shape, dtype=torch.bfloat16)
            g = torch.randn(shape[-1], dtype=torch.bfloat16) if gdtype else None
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi4
            cfg.fp32_dest_acc_en = fp32acc
            cfg.math_approx_mode = False
            try:
                ti = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
                tg = (
                    ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=gdtype, layout=ttnn.TILE_LAYOUT, device=device)
                    if gdtype
                    else None
                )
                out = rms_norm(ti, gamma=tg, compute_kernel_config=cfg)
                got = ttnn.to_torch(out)
                exp = ref(x, g)
                p = pcc(got, exp)
                err = ((got.float() - exp).pow(2).mean().sqrt() / exp.std()).item()
                print(f"  {shape}: PCC={p:.6f} rms_rel={err:.4f}")
            except Exception as e:
                print(f"  {shape}: EXC {type(e).__name__}: {str(e)[:250]}")
finally:
    ttnn.close_device(device)
