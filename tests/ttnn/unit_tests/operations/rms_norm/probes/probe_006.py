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


def cfgof(acc, fid=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def run(device, shape, dtype, gdtype, layout, glayout, acc, poison=None, fid=ttnn.MathFidelity.HiFi4):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16) if gdtype else None
    ti = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=gdtype, layout=glayout, device=device) if gdtype else None
    if poison is not None:
        if layout == ttnn.TILE_LAYOUT:
            ti = ttnn.fill_implicit_tile_padding(ti, poison)
        if tg is not None and glayout == ttnn.TILE_LAYOUT:
            tg = ttnn.fill_implicit_tile_padding(tg, poison)
    out = rms_norm(ti, gamma=tg, compute_kernel_config=cfgof(acc, fid))
    got = ttnn.to_torch(out)
    exp = ref(x, g)
    return pcc(got, exp), ((got.float() - exp).pow(2).mean().sqrt() / exp.std()).item()


T, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
device = ttnn.open_device(device_id=0)
try:
    print("== fp32 + accFalse must be excluded")
    try:
        run(device, (1, 1, 32, 64), ttnn.float32, ttnn.bfloat16, T, T, False)
        print("  excl: NOT REFUSED -- BUG")
    except Exception as e:
        print("  excl:", type(e).__name__, str(e)[:120])

    print("== bfp8 gamma on non-aligned shapes (bf16 in, TILE)")
    for s in [(1, 1, 32, 50), (1, 1, 64, 17), (1, 1, 17, 64), (1, 1, 17, 50), (2, 1, 100, 47), (1, 1, 32, 4095)]:
        for acc in (True, False):
            try:
                p, e = run(device, s, ttnn.bfloat16, ttnn.bfloat8_b, T, T, acc)
                print(f"  {s} acc={acc}: PCC={p:.6f} rms_rel={e:.4f}")
            except Exception as ex:
                print(f"  {s} acc={acc}: EXC {type(ex).__name__}: {str(ex)[:200]}")

    print("== bfp8 gamma with ROW_MAJOR input")
    for s in [(1, 1, 32, 64), (1, 1, 32, 50), (1, 1, 17, 64)]:
        try:
            p, e = run(device, s, ttnn.bfloat16, ttnn.bfloat8_b, RM, T, False)
            print(f"  {s}: PCC={p:.6f} rms_rel={e:.4f}")
        except Exception as ex:
            print(f"  {s}: EXC {type(ex).__name__}: {str(ex)[:200]}")

    print("== bfp8 input, wide W (cross-core combine G>1)")
    for s in [(1, 1, 32, 16384), (1, 1, 32, 32768), (1, 1, 64, 12288), (1, 1, 160, 11008)]:
        for acc in (True, False):
            try:
                p, e = run(device, s, ttnn.bfloat8_b, ttnn.bfloat8_b, T, T, acc)
                print(f"  {s} acc={acc}: PCC={p:.6f} rms_rel={e:.4f}")
            except Exception as ex:
                print(f"  {s} acc={acc}: EXC {type(ex).__name__}: {str(ex)[:200]}")

    print("== pad_poison at accFalse (HiFi2), bf16 TILE")
    for s in [(1, 1, 32, 40), (1, 1, 32, 72), (1, 1, 32, 136), (1, 1, 32, 200), (1, 1, 224, 72), (1, 1, 40, 40)]:
        try:
            p, e = run(device, s, ttnn.bfloat16, ttnn.bfloat16, T, T, False, poison=1000.0, fid=ttnn.MathFidelity.HiFi2)
            print(f"  {s}: PCC={p:.6f} rms_rel={e:.4f}")
        except Exception as ex:
            print(f"  {s}: EXC {type(ex).__name__}: {str(ex)[:200]}")

    print("== RM in/out at accFalse (resilience corner)")
    for s in [(1, 1, 32, 50), (1, 1, 17, 64), (1, 1, 333, 1000), (99991, 64), (1, 1, 3232, 96)]:
        try:
            p, e = run(device, s, ttnn.bfloat16, ttnn.bfloat16, RM, RM, False)
            print(f"  {s}: PCC={p:.6f} rms_rel={e:.4f}")
        except Exception as ex:
            print(f"  {s}: EXC {type(ex).__name__}: {str(ex)[:200]}")

    print("== fp32 in at accTrue with bfp8 gamma, non-aligned")
    for s in [(1, 1, 32, 50), (1, 1, 17, 64)]:
        try:
            p, e = run(device, s, ttnn.float32, ttnn.bfloat8_b, T, T, True)
            print(f"  {s}: PCC={p:.6f} rms_rel={e:.4f}")
        except Exception as ex:
            print(f"  {s}: EXC {type(ex).__name__}: {str(ex)[:200]}")
finally:
    ttnn.close_device(device)
