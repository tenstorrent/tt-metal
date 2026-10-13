# Bit identity on the UniAD PanSegformerHead calls (#59423): ttnn.topk of (1, 900) at k 100 with stable=True (the
# single-core rank-stamped engine from K 128 down, the comparator on main) and ttnn.sort of (1, 101), descending,
# stable. Inputs: sigmoid of bf16 scores at several scales (long runs of exact ties near 1.0) and the ab classes' rows.
# usage: python ab_dump_uniad.py <outdir>
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


inputs = {}
for seed in (0, 1, 2):
    for scale in (1.0, 4.0, 16.0):
        g = torch.Generator().manual_seed(seed)
        inputs[f"sig_s{seed}_x{int(scale)}"] = torch.sigmoid(torch.randn((300, 3), generator=g) * scale).to(torch.bfloat16).reshape(1, -1)
for cname, x in classes().items():
    for r in (0, 7):
        inputs[f"{cname}_r{r}"] = x.reshape(32, 2048)[r, :900].reshape(1, 900).contiguous()

dev = ttnn.open_device(device_id=0)
try:
    for name, x in inputs.items():
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for k in (100, 128, 65):
            for largest in (True, False):
                run(f"uniadtopk_{name}_k{k}_l{int(largest)}", lambda: ttnn.topk(xt, k=k, dim=-1, largest=largest, sorted=True, stable=True))
        xs = ttnn.from_torch(x[:, :101].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for desc in (True, False):
            run(f"uniadsort_{name}_d{int(desc)}", lambda: ttnn.sort(xs, dim=-1, descending=desc, stable=True))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
