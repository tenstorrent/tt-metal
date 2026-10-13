# Issue resolution, bit identity: run the ops on the TopK network on fixed inputs (every bf16 bit pattern, ties, signed
# zeros, special values, random) and save every output, so two variants of the device code can be compared bit for bit.
# usage: python ab_dump.py <outdir>
import os
import sys

import torch
if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")
import ttnn

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def bf16_from_bits(bits):
    return bits.to(torch.int16).view(torch.bfloat16)


def all_patterns(perm_seed):
    bits = torch.arange(65536, dtype=torch.int32)
    if perm_seed == "rev":
        bits = bits.flip(0)
    elif perm_seed is not None:
        g = torch.Generator().manual_seed(perm_seed)
        bits = bits[torch.randperm(65536, generator=g)]
    return bf16_from_bits(bits).reshape(1, 1, 32, 2048)


def classes():
    g = torch.Generator().manual_seed(7)
    out = {
        "allpat": all_patterns(None),
        "allpat_rev": all_patterns("rev"),
        "allpat_perm": all_patterns(11),
        "ties4": torch.randint(-1, 3, (1, 1, 32, 2048), generator=g).to(torch.bfloat16),
        "randn": torch.randn((1, 1, 32, 2048), generator=g).to(torch.bfloat16),
    }
    z = torch.zeros((1, 1, 32, 2048), dtype=torch.bfloat16)
    z[..., 1::2] = -0.0
    z[..., ::7] = 1.0
    z[..., ::11] = float("inf")
    z[..., ::13] = -float("inf")
    out["signed_zero_inf"] = z
    return out


def finite(x):
    # The ops that compute on the values (softmax, gates) get the finite patterns only.
    x = x.float()
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x)).clamp(-60, 60)


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


dev = ttnn.open_device(device_id=0)
one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
try:
    cls = classes()
    for cname, x in cls.items():
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for k in (8, 32, 64):
            for largest in (True, False):
                for stable in (False, True):
                    run(f"topk_{cname}_k{k}_l{int(largest)}_s{int(stable)}",
                        lambda: ttnn.topk(xt, k=k, dim=-1, largest=largest, sorted=True, stable=stable))
                    run(f"topk1c_{cname}_k{k}_l{int(largest)}_s{int(stable)}",
                        lambda: ttnn.topk(xt, k=k, dim=-1, largest=largest, sorted=True, stable=stable, sub_core_grids=one_core))
        xf = ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        for k in (32, 64):
            run(f"topkfp32_{cname}_k{k}", lambda: ttnn.topk(xf, k=k, dim=-1, largest=True, sorted=True))
            run(f"topkfp32_1c_{cname}_k{k}", lambda: ttnn.topk(xf, k=k, dim=-1, largest=True, sorted=True, sub_core_grids=one_core))
        for desc in (True, False):
            for stable in (False, True):
                run(f"sort_{cname}_d{int(desc)}_s{int(stable)}", lambda: ttnn.sort(xt, dim=-1, descending=desc, stable=stable))
        # topk_large_indices: K' 1024 (k 1024 over 2048 per row) and K' 2048 (k 2048 over 8 rows of 8192)
        xr = ttnn.from_torch(x.reshape(32, 2048), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        run(f"tli_{cname}_k1024", lambda: ttnn.experimental.topk_large_indices(xr, k=1024))
        xr8 = ttnn.from_torch(x.reshape(8, 8192), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        run(f"tli_{cname}_k2048", lambda: ttnn.experimental.topk_large_indices(xr8, k=2048))
        # ops that compute on the values
        f = finite(x)
        m = f[..., :64].to(torch.bfloat16).clone()
        m[..., 8:] = 0
        em = torch.zeros((1, 1, 1, 64), dtype=torch.bfloat16)
        em[..., 8:] = float("-inf")
        tm = torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)
        tm[..., 2:] = float("-inf")
        t = lambda a: ttnn.from_torch(a, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        run(f"moe_{cname}", lambda: ttnn.moe(t(m), t(em), t(tm), 32))
        v = ttnn.from_torch(f[..., :256].to(torch.bfloat16), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        i = ttnn.from_torch(torch.arange(0, 256, dtype=torch.int32).expand((1, 1, 32, 256)), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        kk = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pp = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tp = ttnn.from_torch(torch.ones(32), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        run(f"sampling_{cname}", lambda: ttnn.sampling(v, i, k=kk, p=pp, temp=tp, seed=2024))
        for dt, dtn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
            sc = f.reshape(1, 1, 256, 256).contiguous()
            bias = torch.linspace(-1, 1, 256).expand(1, 1, 256, 256).contiguous()
            s_in = ttnn.from_torch(sc, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            b_in = ttnn.from_torch(bias, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            bs = ttnn.from_torch(torch.zeros_like(sc), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            run(f"mgt_{cname}_{dtn}", lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                s_in, b_in, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
                route_scale=0.5, stable_sort=False, epsilon=1e-20, score_func="sigmoid", padding_config=None,
                biased_scores=bs))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
