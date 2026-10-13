# Bit identity with NaN and infinities kept: ttnn.sampling (W 256 and 64) and the DeepSeek prefill gate's stable path
# (moe_grouped_topk with stable_sort) on every class of ab_classes.py unfiltered, plus rows made of NaNs, infinities and
# ties around them.
# usage: python ab_dump_nan.py <outdir>
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


def special_classes():
    g = torch.Generator().manual_seed(23)
    bits = torch.tensor([0x7FC0, 0x7F81, 0x7FFF, 0xFFC0, 0xFF81, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000],
                        dtype=torch.int32)
    pick = torch.randint(0, bits.numel(), (32, 2048), generator=g)
    nan_inf = bits[pick].to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 2048)
    # mostly finite rows with a few NaNs and infinities in each
    base = torch.randn((32, 2048), generator=g)
    m = torch.rand((32, 2048), generator=g)
    base[m < 0.02] = float("nan")
    base[(m >= 0.02) & (m < 0.03)] = float("inf")
    base[(m >= 0.03) & (m < 0.04)] = -float("inf")
    sparse = base.to(torch.bfloat16).reshape(1, 1, 32, 2048)
    return {"nan_inf_mix": nan_inf, "sparse_nan_inf": sparse}


dev = ttnn.open_device(device_id=0)
try:
    cls = dict(classes())
    cls.update(special_classes())
    for cname, x in cls.items():
        for w in (256, 64):
            v = ttnn.from_torch(x[..., :w].contiguous(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            i = ttnn.from_torch(torch.arange(0, w, dtype=torch.int32).expand((1, 1, 32, w)), device=dev, dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT)
            kk = ttnn.from_torch(torch.tensor([1, 10, 15, 20, 25, 30, 32, 5] * 4), device=dev, dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT)
            pp = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 0.1, 0.8] * 4), device=dev, dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT)
            tp = ttnn.from_torch(torch.tensor([1.0, 0.5, 2.0, 1.0] * 8), device=dev, dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT)
            for seed in (2024, 7):
                run(f"samplingnan_{cname}_w{w}_s{seed}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=seed))
        sc = x.float().reshape(1, 1, 256, 256).contiguous()
        bias = torch.linspace(-1, 1, 256).expand(1, 1, 256, 256).contiguous()
        for dt, dtn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
            for func in ("sigmoid", "sqrtsoftplus"):
                s_in = ttnn.from_torch(sc, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                b_in = ttnn.from_torch(bias, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                bs = ttnn.from_torch(torch.zeros_like(sc), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                run(f"mgtnan_{cname}_{dtn}_{func}", lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                    s_in, b_in, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
                    route_scale=0.5, stable_sort=True, epsilon=1e-20, score_func=func, padding_config=None,
                    biased_scores=bs))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
