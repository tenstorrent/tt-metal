# Bit identity of topk_large_indices on the K' 2048 segmented body (rows of more than 32 chunks), whose later segments
# are split across the threads with the accumulator held in L1: every class of ab_classes.py tiled to each shape, NaN
# and infinities kept. Shapes: full segments, a last segment of one chunk, of a few chunks with a tail, the decode width,
# valid_length rows above and below 32 chunks in a wide program, and k 1536.
# usage: python ab_dump_tli2.py <outdir>
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


# (rows, width, k, valid_length)
SHAPES = [
    (2, 131072, 2048, None),   # 64 chunks: two full segments
    (2, 135168, 2048, None),   # 66 chunks: 32, 32, 2
    (2, 133000, 2048, None),   # 65 chunks, tail 1928: last segment one partial chunk (single thread)
    (2, 140000, 2048, None),   # 69 chunks, tail 736: last segment of five
    (1, 524288, 2048, None),   # 256 chunks: the decode width, eight segments
    (2, 262144, 2048, 140000), # 128-chunk program, 69 valid chunks
    (2, 262144, 2048, 60000),  # 128-chunk program, 30 valid chunks (one split row)
    (2, 102400, 1536, None),   # k 1536, 50 chunks: 32, 18
]

dev = ttnn.open_device(device_id=0)
try:
    for cname, x in classes().items():
        flat = x.reshape(-1)

        def tiled(n):
            reps = (n + flat.numel() - 1) // flat.numel()
            return flat.repeat(reps)[:n]

        for rows, n, k, vl in SHAPES:
            xr = ttnn.from_torch(tiled(rows * n).reshape(rows, n), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=dev)
            kw = {"valid_length": vl} if vl is not None else {}
            run(f"tli2_{cname}_{rows}x{n}_k{k}_v{vl}", lambda: ttnn.experimental.topk_large_indices(xr, k=k, **kw))
            xr.deallocate()
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
