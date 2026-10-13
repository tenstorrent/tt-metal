# Bit identity, the paths ab_dump.py leaves out: one-core top-k at k 128 (bf16, the comparator in a 16-bit DEST when
# stable), one-core 32 x 65600 (32-bit indices), and moe_grouped_topk with stable_sort=True.
# usage: python ab_dump_extra.py <outdir>
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


def wide_classes():
    # every row holds all 65536 bf16 bit patterns (its own order) and 64 copies of 1.0; and small integers with ties
    rows = []
    for r in range(32):
        g = torch.Generator().manual_seed(100 + r)
        bits = torch.randperm(65536, generator=g).to(torch.int32)
        rows.append(torch.cat([bits.to(torch.int16).view(torch.bfloat16), torch.ones(64, dtype=torch.bfloat16)]))
    g = torch.Generator().manual_seed(9)
    return {
        "allpat_rows": torch.stack(rows).reshape(1, 1, 32, 65600),
        "ties4w": torch.randint(-1, 3, (1, 1, 32, 65600), generator=g).to(torch.bfloat16),
    }


def finite(x):
    x = x.float()
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x)).clamp(-60, 60)


dev = ttnn.open_device(device_id=0)
one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
try:
    for cname, x in classes().items():
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for largest in (True, False):
            for stable in (False, True):
                run(f"topk1c_{cname}_k128_l{int(largest)}_s{int(stable)}",
                    lambda: ttnn.topk(xt, k=128, dim=-1, largest=largest, sorted=True, stable=stable, sub_core_grids=one_core))
        f = finite(x)
        for dt, dtn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
            for func in ("sigmoid", "sqrtsoftplus"):
                sc = f.reshape(1, 1, 256, 256).contiguous()
                bias = torch.linspace(-1, 1, 256).expand(1, 1, 256, 256).contiguous()
                s_in = ttnn.from_torch(sc, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                b_in = ttnn.from_torch(bias, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                bs = ttnn.from_torch(torch.zeros_like(sc), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                run(f"mgts_{cname}_{dtn}_{func}", lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                    s_in, b_in, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
                    route_scale=0.5, stable_sort=True, epsilon=1e-20, score_func=func, padding_config=None,
                    biased_scores=bs))
    # topk_large_indices at K' 512 (rtdetr, the MSA rows, the classic body of the wide sampling route) and the routed
    # ttnn.topk: every class's 65536 values tiled to each shape
    for cname, x in classes().items():
        flat = x.reshape(-1)
        def tiled(n):
            reps = (n + flat.numel() - 1) // flat.numel()
            return flat.repeat(reps)[:n]
        for rows, n, k, vl in ((1, 8400, 304, None), (1280, 480, 16, 440), (64, 8192, 16, None), (32, 128256, 64, None)):
            xr = ttnn.from_torch(tiled(rows * n).reshape(rows, n), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            kw = {"valid_length": vl} if vl is not None else {}
            run(f"tli512_{cname}_{rows}x{n}_k{k}", lambda: ttnn.experimental.topk_large_indices(xr, k=k, **kw))
        xt = ttnn.from_torch(tiled(32 * 128256).reshape(1, 1, 32, 128256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        run(f"topkroute_{cname}_32x128256_k64", lambda: ttnn.topk(xt, k=64, dim=-1, largest=True, sorted=True))
    # ttnn.sampling at the production widths (W 256 and 64), several seeds; finite inputs (denormals and both zeros
    # kept) and the same with the -inf of masked vocabulary kept
    for cname, x in classes().items():
        xf = x.float()
        fin = torch.where(torch.isfinite(xf), xf, torch.zeros_like(xf)).clamp(-60, 60).to(torch.bfloat16)
        ninf = torch.where(torch.isnan(xf) | (xf == float("inf")), torch.zeros_like(xf), xf).to(torch.bfloat16)
        for vname, vals in (("fin", fin), ("ninf", ninf)):
            for w in (256, 64):
                v = ttnn.from_torch(vals[..., :w].contiguous(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                i = ttnn.from_torch(torch.arange(0, w, dtype=torch.int32).expand((1, 1, 32, w)), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                kk = ttnn.from_torch(torch.tensor([1, 10, 15, 20, 25, 30, 32, 5] * 4), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                pp = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 0.1, 0.8] * 4), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                tp = ttnn.from_torch(torch.tensor([1.0, 0.5, 2.0, 1.0] * 8), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                for seed in (2024, 1, 7, 99):
                    run(f"samplingx_{cname}_{vname}_w{w}_s{seed}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=seed))
    for cname, x in wide_classes().items():
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for largest in (True, False):
            for stable in (False, True):
                run(f"topk1c_{cname}_k32_l{int(largest)}_s{int(stable)}",
                    lambda: ttnn.topk(xt, k=32, dim=-1, largest=largest, sorted=True, stable=stable, sub_core_grids=one_core))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
