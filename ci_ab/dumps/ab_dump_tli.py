# Bit identity of topk_large_indices with the split extended (K' 2048 rows of up to 32 chunks; K' 512 rows of any width,
# the Classic body's included, in segments of 32): every class of ab_classes.py tiled to each shape, NaN and infinities
# kept. Shapes cover one-segment rows, several segments, a one-chunk last segment (the single-thread fallback),
# valid_length rows and the routed ttnn.topk of the wide sampling row.
# usage: python ab_dump_tli.py <outdir>
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
    (32, 128256, 64, None),   # K' 512 Classic, 251 chunks: 8 segments, last of 27
    (4, 32768, 64, None),     # K' 512 Classic, 64 chunks: two full segments
    (4, 16896, 64, None),     # K' 512 Classic, 33 chunks: one-chunk last segment, single-thread body
    (8, 128256, 64, 20000),   # K' 512 Classic program, 40 valid chunks
    (8, 128256, 64, 3000),    # K' 512 Classic program, 6 valid chunks: one segment
    (1, 8400, 304, None),     # K' 512 fused end to end (rtdetr)
    (64, 8192, 16, None),     # K' 512 fused end to end (MSA)
    (64, 1024, 16, 440),      # K' 512, two-chunk program, one valid chunk
    (32, 51200, 1536, None),  # K' 2048, 25 chunks (prefill)
    (2, 102400, 1536, 56320), # K' 2048, 50-chunk program, 28 valid chunks (bounded cache)
    (4, 65536, 2048, None),   # K' 2048, 32 chunks
    (4, 4096, 1536, None),    # K' 2048, 2 chunks
    (2, 67584, 2048, None),   # K' 2048, 33 chunks: segmented single-thread body
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
            run(f"tli_{cname}_{rows}x{n}_k{k}_v{vl}", lambda: ttnn.experimental.topk_large_indices(xr, k=k, **kw))
            xr.deallocate()
        xt = ttnn.from_torch(tiled(32 * 128256).reshape(1, 1, 32, 128256), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        run(f"topkroute_{cname}_32x128256_k64", lambda: ttnn.topk(xt, k=64, dim=-1, largest=True, sorted=True))
        xt.deallocate()
finally:
    torch.save(results, os.path.join(OUT, "results.pt"))
    print("DUMP DONE", len(results), "ok", len(fails), "failed", flush=True)
    ttnn.close_device(dev)
