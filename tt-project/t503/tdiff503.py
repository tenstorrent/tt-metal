"""t503: per-tensor diff of the stale vs Aug-17 LTX-2.5 Gemma-4 encoder files."""
import sys, torch
from safetensors import safe_open

a, b = sys.argv[1], sys.argv[2]
with safe_open(a, "pt") as fa, safe_open(b, "pt") as fb:
    ka, kb = set(fa.keys()), set(fb.keys())
    print("only_old", sorted(ka - kb)[:20], "only_new", sorted(kb - ka)[:20])
    same = diff = 0
    groups = {}
    for k in sorted(ka & kb):
        x, y = fa.get_tensor(k), fb.get_tensor(k)
        if x.shape != y.shape or x.dtype != y.dtype:
            print("SHAPE", k, x.shape, y.shape, x.dtype, y.dtype)
            diff += 1
            continue
        if torch.equal(x, y):
            same += 1
            continue
        diff += 1
        if x.is_floating_point():
            xd, yd = x.double().flatten(), y.double().flatten()
            rel = float((xd - yd).norm() / xd.norm().clamp_min(1e-30))
            print(f"DIFF {k} {tuple(x.shape)} rel={rel:.4g}")
        else:
            print(f"DIFF {k} {tuple(x.shape)} non-float")
    print(f"same={same} diff={diff}")
