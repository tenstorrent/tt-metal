# Bit identity of ttnn.sort(stable=True) on bf16 TILE inputs (the single-core sort kernel, W up to 2048): every bf16
# pattern at several widths, both directions, plus tie-heavy and signed-zero/denormal-heavy rows.
# usage: python ab_dump_sort.py <outdir>
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


def bf16_bits(bits):
    return bits.to(torch.int16).view(torch.bfloat16)


def classes(w):
    g = torch.Generator().manual_seed(w)
    flat = {
        "allpat": torch.arange(65536, dtype=torch.int32),
        "allpat_perm": torch.randperm(65536, generator=g).to(torch.int32),
    }
    out = {}
    for n, bits in flat.items():
        rows = (65536 + w - 1) // w
        reps = rows * w
        out[n] = bf16_bits(bits.repeat(2)[:reps]).reshape(rows, w)
    out["ties"] = torch.randint(-1, 3, (64, w), generator=g).to(torch.bfloat16)
    # +0, -0, the 254 denormals and +-1, many of each per row
    small = torch.cat([torch.arange(0, 128), torch.arange(0x8000, 0x8080), torch.tensor([0x3F80, 0xBF80])]).to(torch.int32)
    pick = torch.randint(0, small.numel(), (64, w), generator=g)
    out["zero_denorm"] = bf16_bits(small[pick])
    out["randn"] = torch.randn((64, w), generator=g).to(torch.bfloat16)
    return out


dev = ttnn.open_device(device_id=0)
try:
    for w in (64, 101, 128, 256, 1000, 2048):
        for cname, x in classes(w).items():
            shapes = [("2d", x)] if w == 101 else []
            shapes.append(("4d", x.reshape(1, 1, *x.shape)))
            for sname, xs in shapes:
                xt = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                for desc in (True, False):
                    run(f"sort_{cname}_w{w}_{sname}_d{int(desc)}", lambda: ttnn.sort(xt, dim=-1, descending=desc, stable=True))
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
