# Bit identity of topk_large_indices on a core's second and third segmented rows (K' 2048, rows of more than 32 chunks,
# the accumulator held in L1 between segments): more rows than cores, every class of ab_classes.py (or the named ones)
# tiled to each shape, NaN and infinities kept.
# usage: python ab_dump_tli5.py <outdir>
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


# (rows, width, k, valid_length, classes or None for all): more rows than the 110 cores, so cores run a second (and
# third) segmented row after a row whose later segments were held in L1.
SHAPES = [
    (256, 140000, 2048, None, None),   # 69 chunks: 32, 32, 5
    (256, 135168, 2048, None, None),   # 66 chunks: 32, 32, 2 (the last segment on one thread)
    (160, 524288, 2048, None, ("allpat", "ties4", "signed_zero_inf")),  # GLM 5.3 decode, 256 chunks
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
            run(f"tli5_{cname}_{rows}x{n}_k{k}_v{vl}", lambda: ttnn.experimental.topk_large_indices(xr, k=k, **kw))
            xr.deallocate()
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
