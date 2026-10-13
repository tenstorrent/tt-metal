# Bit identity on the paths the cross-family audit listed as undumped: one-core stable ttnn.topk on bfp8 and bfp4
# inputs (the rank-stamped engine for a uint16 index output, the comparator on main) at k 8, 32, 64, 100 and 128,
# largest and smallest, and ttnn.experimental.deepseek_grouped_gate on every class (scores and bias as bf16).
# usage: python ab_dump_audit.py <outdir>
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


dev = ttnn.open_device(device_id=0)
one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
try:
    for cname, x in classes().items():
        f = finite(x).to(torch.bfloat16)
        for dt, dtn in ((ttnn.bfloat8_b, "bfp8"), (ttnn.bfloat4_b, "bfp4")):
            xt = ttnn.from_torch(f, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            for k in (8, 32, 64, 100, 128):
                for largest in (True, False):
                    run(f"topk1c_{dtn}_{cname}_k{k}_l{int(largest)}_s1",
                        lambda: ttnn.topk(xt, k=k, dim=-1, largest=largest, sorted=True, stable=True, sub_core_grids=one_core))
        sc = f.reshape(1, 1, 256, 256).contiguous()
        bias = torch.linspace(-1, 1, 256, dtype=torch.float32).to(torch.bfloat16).expand(1, 1, 256, 256).contiguous()
        s_in = ttnn.from_torch(sc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        b_in = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        run(f"dsgate_{cname}", lambda: ttnn.experimental.deepseek_grouped_gate(
            s_in, b_in, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
            route_scale=1.0, epsilon=1e-20))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
