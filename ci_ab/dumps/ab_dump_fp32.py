# Issue resolution, bit identity of the fp32 paths: fp32 keys with every low mantissa bit in play (random 32-bit
# patterns, every sign and exponent with edge mantissas, NaN payloads, denormals, neighbours one ulp apart).
# usage: python ab_dump_fp32.py <outdir>
import os
import sys

import torch
if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")
import ttnn

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
N = 32 * 2048


def from_bits(bits):
    return bits.to(torch.int64).to(torch.int32).view(torch.float32).reshape(1, 1, 32, 2048)


def classes():
    g = torch.Generator().manual_seed(23)
    out = {"rand32": from_bits(torch.randint(0, 2**32, (N,), generator=g, dtype=torch.int64))}
    se = torch.arange(512, dtype=torch.int64) << 23
    mant = torch.tensor([0, 1, 2, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFE, 0x7FFFFF], dtype=torch.int64)
    edge = (se[:, None] | mant[None, :]).reshape(-1)
    edge = torch.cat([edge, torch.randint(0, 2**32, (N - edge.numel(),), generator=g, dtype=torch.int64)])
    out["edges"] = from_bits(edge[torch.randperm(N, generator=g)])
    base = torch.randint(0x3F000000, 0x40800000, (N // 4,), generator=g, dtype=torch.int64)
    near = torch.stack([base, base + 1, base ^ 0x80000000, base], 1).reshape(-1)
    out["ulp_ties"] = from_bits(near[torch.randperm(N, generator=g)])
    return out


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
    for cname, x in classes().items():
        xf = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        for k in (8, 32, 64):
            for largest in (True, False):
                for stable in (False, True):
                    run(f"f32_{cname}_k{k}_l{int(largest)}_s{int(stable)}",
                        lambda: ttnn.topk(xf, k=k, dim=-1, largest=largest, sorted=True, stable=stable))
                    run(f"f32_1c_{cname}_k{k}_l{int(largest)}_s{int(stable)}",
                        lambda: ttnn.topk(xf, k=k, dim=-1, largest=largest, sorted=True, stable=stable, sub_core_grids=one_core))
        for desc in (True, False):
            for stable in (False, True):
                run(f"f32sort_{cname}_d{int(desc)}_s{int(stable)}", lambda: ttnn.sort(xf, dim=-1, descending=desc, stable=stable))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
