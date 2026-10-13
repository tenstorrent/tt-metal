# Bit identity of topk_large_indices at K' 1024 (k in 513..1024): rows of 2 to 32 chunks (the math/pack split), one
# chunk and 33 chunks (single thread) as controls, valid_length tails, more rows than cores, every class of
# ab_classes.py (or the named ones) tiled to each shape, NaN and infinities kept. usage: python ab_dump_tli1024.py <outdir>
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


# (rows, width, k, valid_length, classes or None for all)
SHAPES = [
    (220, 2048, 1024, None, None),    # 2 chunks, two rows per core
    (220, 3328, 1024, 3310, None),    # 4 chunks, tail 238 (DeepSeek V4 Pro CSA indexer, 13240 tokens / 4)
    (130, 8192, 1024, None, None),    # 8 chunks
    (64, 5000, 640, None, None),      # k 640 snaps to K' 1024: 5 chunks, tail 904
    (120, 32768, 1024, None, ("allpat", "ties4", "signed_zero_inf", "randn")),  # 32 chunks, the split's longest row
    (16, 65536, 1024, 30000, ("allpat", "ties4")),  # 64-chunk program, 30 valid chunks (one split row)
    (120, 33792, 1024, None, ("allpat", "ties4")),  # 33 chunks: single-thread segmented body (control)
    (220, 1024, 1024, None, ("allpat", "ties4")),   # 1 chunk: no split (control)
]

dev = ttnn.open_device(device_id=0)
try:
    for cname, x in classes().items():
        flat = x.reshape(-1)

        def tiled(n):
            reps = (n + flat.numel() - 1) // flat.numel()
            return flat.repeat(reps)[:n]

        for rows, n, k, vl, only in SHAPES:
            if only is not None and cname not in only:
                continue
            xr = ttnn.from_torch(tiled(rows * n).reshape(rows, n), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=dev)
            kw = {"valid_length": vl} if vl is not None else {}
            run(f"tli1024_{cname}_{rows}x{n}_k{k}_v{vl}", lambda: ttnn.experimental.topk_large_indices(xr, k=k, **kw))
            xr.deallocate()
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
