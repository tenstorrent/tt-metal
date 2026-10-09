import json

from safetensors import safe_open


def idx(p):
    m = json.load(open(p + "/model.safetensors.index.json"))["weight_map"]
    return m


A = "/home/ttuser/atupe/models/Qwen3.8-27B"
B = "/home/runara/models/Qwen3.6-27B"
ia, ib = idx(A), idx(B)
print("keys equal", set(ia) == set(ib), len(ia))
import collections

res = []
dt = collections.Counter()
hA = {}
hB = {}


def get(p, m, k, h):
    f = m[k]
    if f not in h:
        h[f] = safe_open(p + "/" + f, "pt")
    return h[f].get_tensor(k)


for k in sorted(ia):
    if "visual" in k:
        continue
    a = get(A, ia, k, hA)
    b = get(B, ib, k, hB)
    dt[(str(a.dtype), str(b.dtype))] += 1
    a = a.float()
    b = b.float()
    rel = ((a - b).norm() / (b.norm() + 1e-9)).item()
    res.append((k, rel, a.abs().max().item(), b.abs().max().item(), a.numel()))
print(dt)
import re

agg = collections.defaultdict(list)
for k, rel, ma, mb, n in res:
    g = re.sub(r"\.\d+\.", ".N.", k)
    agg[g].append((rel, ma, mb))
for g, v in sorted(agg.items()):
    import statistics

    print(
        f"{g:70s} n={len(v):3d} rel_diff mean {statistics.mean(x[0] for x in v):.3f} max {max(x[0] for x in v):.3f} | absmax38/36 max-ratio {max(x[1]/max(x[2],1e-9) for x in v):.2f} (38max {max(x[1] for x in v):.2f}, 36max {max(x[2] for x in v):.2f})"
    )
