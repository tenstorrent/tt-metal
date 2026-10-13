# Bit identity of the DeepSeek grouped gate and the GPT-OSS fused router (they run the bitonic local sort): random,
# tie-heavy and signed-zero scores at the unit tests' shapes. usage: python ab_dump_gates.py <outdir>
import os
import sys

import torch
if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")
import ttnn

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


def tile(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def score_classes(shape, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(shape, generator=g)
    ties = (torch.randint(0, 6, shape, generator=g).float() / 8.0)
    zeros = torch.where(torch.rand(shape, generator=g) < 0.5, torch.zeros(shape), -torch.zeros(shape))
    zeros[torch.rand(shape, generator=g) < 0.1] = 1.0
    return {"rand": torch.sigmoid(r), "ties": ties, "zeros": zeros}


row = os.environ.get("DUMP_ROW_DISPATCH") == "1"
kw = {"dispatch_core_config": ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)} if row else {}
dev = ttnn.open_device(device_id=0, **kw)
try:
    if not row:
        for nb, bs, sl in ((1, 1, 1), (1, 1, 33), (1, 8, 512), (7, 7, 81), (1, 1, 8192)):
            shape = (nb, bs, sl, 256)
            for cname, sc in score_classes(shape, 5 + sl).items():
                bias = torch.randn(shape, generator=torch.Generator().manual_seed(sl)) if cname == "rand" else torch.zeros(shape)
                s_t, b_t = tile(sc.to(torch.bfloat16)), tile(bias.to(torch.bfloat16))
                run(f"grouped_gate_{nb}x{bs}x{sl}_{cname}", lambda: ttnn.experimental.deepseek_grouped_gate(
                    s_t, b_t, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
                    route_scale=2.5, epsilon=1e-20))
    else:
        for B, K in ((32, 2880), (1, 2880), (9, 2880), (31, 2880), (32, 64), (32, 4096)):
            for seed in (42, 99):
                g = torch.Generator().manual_seed(seed)
                x = tile(torch.randn((B, K), generator=g).to(torch.bfloat16))
                w = tile((torch.randn((K, 128), generator=g) * 0.02).to(torch.bfloat16))
                b = tile(torch.randn((1, 128), generator=g).expand(B, 128).contiguous().to(torch.bfloat16))
                run(f"router_gpt_{B}x{K}_s{seed}", lambda: ttnn.experimental.topk_router_gpt(x, weight_tensor=w, bias_tensor=b, k=4, num_experts=128))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
