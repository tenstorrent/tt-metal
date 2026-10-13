# Scratch (not for merge): ttnn.topk and ttnn.sort on every bf16 bit pattern, outputs saved per arm for allpat_cmp.py.
import os, sys
import torch
import ttnn

OUT = sys.argv[1]


def allpat(rev):
    bits = torch.arange(65536, dtype=torch.int32)
    if rev:
        bits = bits.flip(0)
    return bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 2048)


dev = ttnn.open_device(device_id=0)
one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
res = {}
try:
    for rev in (False, True):
        x = allpat(rev)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for k in (8, 32, 64):
            for largest in (True, False):
                for one in (False, True):
                    name = f"topk{'1c' if one else ''}_{'rev' if rev else 'fwd'}_k{k}_l{int(largest)}"
                    kw = dict(sub_core_grids=one_core) if one else {}
                    v, i = ttnn.topk(xt, k=k, dim=-1, largest=largest, sorted=True, stable=False, **kw)
                    res[name] = (ttnn.to_torch(v).clone(), ttnn.to_torch(i).clone())
        for desc in (True, False):
            v, i = ttnn.sort(xt, dim=-1, descending=desc, stable=False)
            res[f"sort_{'rev' if rev else 'fwd'}_d{int(desc)}"] = (ttnn.to_torch(v).clone(), ttnn.to_torch(i).clone())
finally:
    torch.save(res, OUT)
    print("ALLPAT DONE", len(res), flush=True)
    ttnn.close_device(dev)
