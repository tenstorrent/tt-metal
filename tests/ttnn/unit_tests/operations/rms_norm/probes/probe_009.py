import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg(fp32_acc, fid=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = fp32_acc
    c.math_approx_mode = False
    return c


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def ref(x, g, eps=1e-6):
    xf = x.float()
    r = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    o = xf / r
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


cases = [
    ("bf8b x, bf8b g", (1, 1, 64, 512), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf8b x, bf16 g", (1, 1, 64, 512), ttnn.bfloat8_b, ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
    ("bf8b x, fp32 g", (1, 1, 64, 512), ttnn.bfloat8_b, ttnn.float32, ttnn.TILE_LAYOUT, True),
    ("bf16 x, bf8b g", (1, 1, 64, 512), ttnn.bfloat16, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf8b x acc=False", (1, 1, 64, 512), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, False),
    ("bf8b x no_gamma", (1, 1, 64, 512), ttnn.bfloat8_b, None, ttnn.TILE_LAYOUT, True),
    ("bf8b x rank2", (128, 512), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf8b x rank3", (4, 128, 512), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf16 x, bf8b g, w_non_align", (1, 1, 32, 72), ttnn.bfloat16, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf16 x, bf8b g, h_non_align", (1, 1, 50, 128), ttnn.bfloat16, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("fp32 x, bf8b g", (1, 1, 64, 512), ttnn.float32, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf8b x RMgamma bf16", (1, 1, 64, 512), ttnn.bfloat8_b, ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
    ("bf8b x STREAM wide", (1, 1, 32, 32768), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
    ("bf8b x big multi-core", (1, 1, 2048, 256), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True),
]

for name, shape, dt, gdt, lay, acc in cases:
    torch.manual_seed(0)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16 if dt != ttnn.float32 else torch.float32)
    g = None if gdt is None else torch.randn(W, dtype=torch.bfloat16 if gdt != ttnn.float32 else torch.float32)
    expected = ref(x, g)
    glay = ttnn.ROW_MAJOR_LAYOUT if "RMgamma" in name else ttnn.TILE_LAYOUT
    tx = ttnn.from_torch(x, dtype=dt, layout=lay, device=device)
    tg = None if g is None else ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=gdt, layout=glay, device=device)
    try:
        out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg(acc))
        a = ttnn.to_torch(out).float()
        err = (a - expected).abs()
        rms = (err.pow(2).mean().sqrt() / expected.std()).item()
        print(f"  {name:32s} PCC={pcc(a,expected):.6f} relRMS={rms:.5f} max={err.max().item():.4g}")
    except Exception as e:
        print(f"  {name:32s} FAIL {type(e).__name__}: {str(e)[:200]}")

ttnn.close_device(device)
