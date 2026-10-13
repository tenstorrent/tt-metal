"""t479: dump vsa_sdpa inputs/outputs (device 0, first T479_HEADS heads) at chosen real-rung calls, then stop.

Real-rung calls are the ones whose local q rows equal T479_ROWS (9472 at the 75776 rung); warm-up calls
are passed through. T479_AT lists the real-call indices to dump (default 0,49,150,199 = step 1 layers 0/49,
step 4 layers 0/49 for 50 layers, no CFG). Each dump goes to $T479_DUMP/call<i>.pt as soon as it is taken;
after the last one the plug raises so pytest tears the mesh down.
"""
import os

import torch
import ttnn

_orig = ttnn.transformer.vsa_sdpa
_state = {"real": 0, "warm": 0}


def _dev0(x):
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def _wrapped(q, k, v, ids, counts, *args, **kwargs):
    out = _orig(q, k, v, ids, counts, *args, **kwargs)
    if q.shape[2] != int(os.environ.get("T479_ROWS", "9472")):
        _state["warm"] += 1
        return out
    i = _state["real"]
    _state["real"] += 1
    at = sorted(int(x) for x in os.environ.get("T479_AT", "0,49,150,199").split(","))
    if i not in at:
        return out
    heads = int(os.environ.get("T479_HEADS", "4"))
    d = {
        "q": _dev0(q)[:, :heads].to(torch.bfloat16),
        "k": _dev0(k)[:, :heads].to(torch.bfloat16),
        "v": _dev0(v)[:, :heads].to(torch.bfloat16),
        "ids": _dev0(ids)[:, :heads],
        "counts": _dev0(counts),
        "out": _dev0(out)[:, :heads].to(torch.bfloat16),
        "kwargs": {a: b for a, b in kwargs.items() if isinstance(b, (int, float, bool, str, type(None)))},
        "shapes": {n: list(t.shape) for n, t in (("q", q), ("k", k), ("ids", ids), ("counts", counts), ("out", out))},
        "skipped_calls": _state["warm"],
        "call": i,
    }
    path = os.path.join(os.environ["T479_DUMP"], f"call{i}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[t479] dumped {path}: {d['shapes']} kwargs {d['kwargs']} warm {_state['warm']}", flush=True)
    if i == at[-1]:
        raise RuntimeError("T479_DUMP_DONE")
    return out


ttnn.transformer.vsa_sdpa = _wrapped
