# Bit identity of ttnn.sampling alone: every sampling case of ab_dump.py (sampling_<class>), ab_dump_extra.py
# (samplingx_*) and ab_dump_nan.py (samplingnan_*), under the same names, plus rows of signed zeros, denormals of both
# signs and a few positives at W 64, 128 and 256 (samplingz_*), where the zeros' order decides the sampled index.
# usage: python ab_dump_samp.py <outdir>
import os
import sys

import torch
if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_classes import classes  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
results = {}
fails = []


def run(name, fn):
    try:
        outs = fn()
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        results[name] = [ttnn.to_torch(o).clone() for o in outs]
        print("CASE", name, "ok", flush=True)
    except Exception as e:  # noqa: BLE001
        fails.append(name)
        print("CASE", name, "FAILED", type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "", flush=True)


def finite(x):
    x = x.float()
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x)).clamp(-60, 60)


def special_classes():
    g = torch.Generator().manual_seed(23)
    bits = torch.tensor([0x7FC0, 0x7F81, 0x7FFF, 0xFFC0, 0xFF81, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000],
                        dtype=torch.int32)
    pick = torch.randint(0, bits.numel(), (32, 2048), generator=g)
    nan_inf = bits[pick].to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 2048)
    base = torch.randn((32, 2048), generator=g)
    m = torch.rand((32, 2048), generator=g)
    base[m < 0.02] = float("nan")
    base[(m >= 0.02) & (m < 0.03)] = float("inf")
    base[(m >= 0.03) & (m < 0.04)] = -float("inf")
    sparse = base.to(torch.bfloat16).reshape(1, 1, 32, 2048)
    return {"nan_inf_mix": nan_inf, "sparse_nan_inf": sparse}


def zero_classes():
    # +0, -0, denormals of both signs, a few 1.0 and -1.0: the top 32 of every width reach the zeros and denormals
    g = torch.Generator().manual_seed(31)
    out = {}
    for name, pool in (
        ("pm0", [0x0000, 0x8000]),
        ("pm0_den", [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0040, 0x8040]),
        ("negden", [0x0000, 0x8000, 0x8001, 0x807F, 0x8040, 0x8010]),
    ):
        pool = torch.tensor(pool, dtype=torch.int32)
        bits = pool[torch.randint(0, pool.numel(), (32, 256), generator=g)]
        sel = torch.rand((32, 256), generator=g)
        bits[sel < 0.03] = 0x3F80
        bits[(sel >= 0.03) & (sel < 0.06)] = 0xBF80
        out[name] = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 256)
    return out


def sampling_inputs(w):
    i = ttnn.from_torch(torch.arange(0, w, dtype=torch.int32).expand((1, 1, 32, w)), device=dev, dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT)
    kk = ttnn.from_torch(torch.tensor([1, 10, 15, 20, 25, 30, 32, 5] * 4), device=dev, dtype=ttnn.uint32,
                         layout=ttnn.ROW_MAJOR_LAYOUT)
    pp = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 0.1, 0.8] * 4), device=dev, dtype=ttnn.bfloat16,
                         layout=ttnn.ROW_MAJOR_LAYOUT)
    tp = ttnn.from_torch(torch.tensor([1.0, 0.5, 2.0, 1.0] * 8), device=dev, dtype=ttnn.bfloat16,
                         layout=ttnn.ROW_MAJOR_LAYOUT)
    return i, kk, pp, tp


dev = ttnn.open_device(device_id=0)
try:
    # ab_dump.py
    for cname, x in classes().items():
        f = finite(x)
        v = ttnn.from_torch(f[..., :256].to(torch.bfloat16), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        i = ttnn.from_torch(torch.arange(0, 256, dtype=torch.int32).expand((1, 1, 32, 256)), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        kk = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pp = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tp = ttnn.from_torch(torch.ones(32), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        run(f"sampling_{cname}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=2024))
    # ab_dump_extra.py
    for cname, x in classes().items():
        xf = x.float()
        fin = torch.where(torch.isfinite(xf), xf, torch.zeros_like(xf)).clamp(-60, 60).to(torch.bfloat16)
        ninf = torch.where(torch.isnan(xf) | (xf == float("inf")), torch.zeros_like(xf), xf).to(torch.bfloat16)
        for vname, vals in (("fin", fin), ("ninf", ninf)):
            for w in (256, 64):
                v = ttnn.from_torch(vals[..., :w].contiguous(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                i, kk, pp, tp = sampling_inputs(w)
                for seed in (2024, 1, 7, 99):
                    run(f"samplingx_{cname}_{vname}_w{w}_s{seed}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=seed))
    # ab_dump_nan.py
    cls = dict(classes())
    cls.update(special_classes())
    for cname, x in cls.items():
        for w in (256, 64):
            v = ttnn.from_torch(x[..., :w].contiguous(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            i, kk, pp, tp = sampling_inputs(w)
            for seed in (2024, 7):
                run(f"samplingnan_{cname}_w{w}_s{seed}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=seed))
    # zeros and denormals
    for cname, x in zero_classes().items():
        for w in (256, 128, 64):
            v = ttnn.from_torch(x[..., :w].contiguous(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            i, kk, pp, tp = sampling_inputs(w)
            for seed in (2024, 1, 7, 99, 5, 13):
                run(f"samplingz_{cname}_w{w}_s{seed}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=seed))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
