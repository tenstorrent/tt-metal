import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def ref(x, g=None, eps=1e-6):
    xf = x.float()
    r = xf / torch.sqrt((xf**2).mean(-1, keepdim=True) + eps)
    if g is not None:
        r = r * g.float().reshape(-1)
    return r


def cfg(fp32_acc, fid=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = fp32_acc
    c.math_approx_mode = False
    return c


CASES = [
    ("bf16 / fp32acc=False / aligned", (1, 1, 64, 128), ttnn.bfloat16, ttnn.bfloat16, False),
    ("bf16 / fp32acc=False / w_non_aligned", (1, 1, 64, 100), ttnn.bfloat16, ttnn.bfloat16, False),
    ("bf16 / fp32acc=False / h_non_aligned", (1, 1, 17, 128), ttnn.bfloat16, ttnn.bfloat16, False),
    ("bf16 / fp32acc=False / wide W=7168", (1, 1, 32, 7168), ttnn.bfloat16, ttnn.bfloat16, False),
    ("bfp8 / fp32acc=True  / aligned", (1, 1, 64, 128), ttnn.bfloat8_b, ttnn.bfloat8_b, True),
    ("bfp8 / fp32acc=False / aligned", (1, 1, 64, 128), ttnn.bfloat8_b, ttnn.bfloat8_b, False),
    ("bfp8 / fp32acc=True  / w_non_aligned", (1, 1, 64, 100), ttnn.bfloat8_b, ttnn.bfloat8_b, True),
    ("bfp8 / fp32acc=True  / h_non_aligned", (1, 1, 17, 128), ttnn.bfloat8_b, ttnn.bfloat8_b, True),
    ("bfp8 / fp32acc=True  / wide W=4096", (1, 1, 32, 4096), ttnn.bfloat8_b, ttnn.bfloat8_b, True),
    ("bf16 in / bfp8 gamma / aligned", (1, 1, 64, 128), ttnn.bfloat16, ttnn.bfloat8_b, True),
    ("fp32 in / bfp8 gamma / aligned", (1, 1, 64, 128), ttnn.float32, ttnn.bfloat8_b, True),
    ("bfp8 in / fp32 gamma / aligned", (1, 1, 64, 128), ttnn.bfloat8_b, ttnn.float32, True),
    ("bfp8 / no gamma / aligned", (1, 1, 64, 128), ttnn.bfloat8_b, None, True),
]

for name, shape, dt, gdt, acc in CASES:
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    x = torch.randn(shape, dtype=tdt)
    if gdt is None:
        g = None
        tg = None
    else:
        gtd = torch.float32 if gdt == ttnn.float32 else torch.bfloat16
        g = torch.randn(shape[-1], dtype=gtd)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=gdt, layout=ttnn.TILE_LAYOUT, device=device)
    tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg(acc))
        got = ttnn.to_torch(out)
        e = ref(x, g)
        print(f"{name:42s} PCC={pcc(got, e):.6f}  maxabs={(got.float()-e).abs().max().item():.4f}", flush=True)
    except Exception as ex:
        print(f"{name:42s} EXC {type(ex).__name__}: {str(ex)[:300]}", flush=True)

ttnn.close_device(device)
