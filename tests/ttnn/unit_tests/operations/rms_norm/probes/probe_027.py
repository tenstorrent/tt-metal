import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config, shard_config


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.to(torch.float32).reshape(-1)
    return o


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


_ML = ttnn.TensorMemoryLayout
MLS = [("H", _ML.HEIGHT_SHARDED), ("W", _ML.WIDTH_SHARDED), ("B", _ML.BLOCK_SHARDED)]
res = []
device = ttnn.open_device(device_id=0)


def run(name, shape, ml, *, dtype=ttnn.bfloat16, gamma=True, poison=None, explicit=None, fp32acc=False):
    tdt = torch.bfloat16
    x = torch.randn(shape, dtype=tdt)
    gm = torch.randn(shape[-1], dtype=tdt) if gamma else None
    try:
        mc = explicit or auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
        tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tg = None
        if gamma:
            tg = ttnn.from_torch(gm.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if poison is not None:
            tx = ttnn.fill_implicit_tile_padding(tx, poison)
            if tg is not None:
                tg = ttnn.fill_implicit_tile_padding(tg, poison)
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = fp32acc
        cfg.math_approx_mode = False
        out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=tx.memory_config())
        got = ttnn.to_torch(out)
        p = pcc(got, ref(x, gm))
        rel = ((got.float() - ref(x, gm).float()).pow(2).mean().sqrt() / ref(x, gm).float().std()).item()
        res.append(
            f"{name} {shape} shard={list(mc.shard_spec.shape)} -> PCC {p:.6f} relRMS {rel:.4f} {'OK' if p>0.99 else 'FAIL'}"
        )
    except Exception as e:
        res.append(f"{name} {shape} -> EXC {type(e).__name__}: {str(e)[:150]}")
    print(res[-1], flush=True)


try:
    # no-gamma
    for tag, ml in MLS:
        run(f"NOGAMMA-{tag}", (1, 1, 256, 512), ml, gamma=False)
    # bfloat8_b
    for tag, ml in MLS:
        run(f"BF8B-{tag}", (1, 1, 256, 512), ml, dtype=ttnn.bfloat8_b)
    # pad_poison (W not tile aligned, padding poisoned)
    for shp in [(1, 1, 32, 40), (1, 1, 32, 72), (1, 1, 224, 72), (1, 1, 40, 40)]:
        for tag, ml in MLS:
            run(f"POISON-{tag}", shp, ml, poison=1000.0)
    # pinned perf geometries
    for shp, ss, gr, ml in [
        ((1, 1, 32, 1024), [32, 128], (8, 1), _ML.WIDTH_SHARDED),
        ((1, 1, 32, 2304), [32, 256], (9, 1), _ML.WIDTH_SHARDED),
        ((1, 1, 32, 5120), [32, 160], (8, 4), _ML.WIDTH_SHARDED),
        ((1, 1, 32, 7168), [32, 256], (7, 4), _ML.WIDTH_SHARDED),
        ((1, 1, 8192, 1024), [1024, 128], (8, 8), _ML.BLOCK_SHARDED),
    ]:
        mc = shard_config(ss, gr, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        run(f"PERF-{ss}-{gr}", shp, ml, explicit=mc)
finally:
    ttnn.close_device(device)
print("=== SUMMARY ===")
for r in res:
    print(r)
